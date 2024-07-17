import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pad_sequence

from predicators.gnn.gnn import MLP, GraphModel
from predicators.gnn.gnn_utils import compute_normalizers, concat_graphs, normalize_graph
from predicators.ml_models import DeviceTrackingModule, _get_torch_device
from predicators.structs import Array

class ResidualAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_sizes):
        """
        Inputs:
            embed_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
        """
        super().__init__()

        # Attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Two-layer MLP
        self.fc = MLP(fc_sizes + [embed_dim], embed_dim)

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # Attention part
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = self.norm1(x)

        # MLP part
        fc_out = self.fc(x)
        x = x + fc_out
        x = self.norm2(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=16):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class EncodeProcessDecode(GraphModel):
    """Encode-process-decode GNN architecture."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        node_encoder_layers: List[int],
        node_model_layers: List[int],
        edge_encoder_layers: List[int],
        edge_model_layers: List[int],
        global_encoder_layers: List[int],
        global_model_layers: List[int],
        global_decoder_layers: List[int],
        num_steps: int,
        use_torch_gpu: bool,
    ) -> None:
        super().__init__(
            dims = [node_dim, edge_dim, global_dim],
            node_encoder_layers = node_encoder_layers,
            node_model_layers = node_model_layers,
            edge_encoder_layers = edge_encoder_layers,
            edge_model_layers = edge_model_layers,
            global_encoder_layers = global_encoder_layers,
            global_model_layers = global_model_layers,
            use_torch_gpu=use_torch_gpu,
        )
        self.global_decoder = MLP(global_decoder_layers, global_model_layers[-1])
        self.num_steps = num_steps
        self.to_common_device()

    def forward(self, graph: Dict) -> List[Dict]:
        """Torch forward model."""
        graph['nodes'] = self.node_encoder(graph['nodes'])
        graph['edges'] = self.edge_encoder(graph['edges'])
        graph['globals'] = self.global_encoder(graph['globals'])

        latent0 = graph
        for _ in range(self.num_steps):
            graph = concat_graphs([latent0, graph], dim=1)
            graph, eg = self.edges(graph)
            graph, ng = self.nodes(graph)
            graph = self.globals(graph, eg, ng)

        return self.global_decoder(graph['globals'])

class Imitation(DeviceTrackingModule):
    def __init__(
        self,

        gnn_node_dim: int,
        gnn_edge_dim: int,
        gnn_global_dim: int,
        gnn_node_layers: List[int],
        gnn_edge_layers: List[int],
        gnn_global_layers: List[int],
        gnn_num_steps: int,

        attention_size: int,
        attention_num_heads: int,
        attention_num_blocks: int,
        attention_fc_sizes: List[int],

        predictor_sizes: List[int],

        lr: float,
        use_torch_gpu: bool,
        params_path: Optional[str] = None,
    ):
        super().__init__(device=_get_torch_device(use_torch_gpu))

        self._gnn_normalizers: Optional[Dict[str, Tuple[Array, Array]]] = None

        # state feature extractor
        self.graph_embedding_net = EncodeProcessDecode(
            node_dim = gnn_node_dim,
            edge_dim = gnn_edge_dim,
            global_dim = gnn_global_dim,
            node_encoder_layers = gnn_node_layers,
            node_model_layers = gnn_node_layers,
            edge_encoder_layers = gnn_edge_layers,
            edge_model_layers = gnn_edge_layers,
            global_encoder_layers = gnn_global_layers,
            global_model_layers = gnn_global_layers,
            global_decoder_layers = gnn_global_layers + [attention_size],
            num_steps = gnn_num_steps,
            use_torch_gpu = use_torch_gpu,
        )

        # attention
        self.positional_encoding = PositionalEncoding(attention_size)
        atten_layers = [ResidualAttentionBlock(attention_size, attention_num_heads, attention_fc_sizes)
                        for _ in range(attention_num_blocks)]
        self.atten_layers = nn.ModuleList(atten_layers)

        # predictor
        self.imit_predictor = MLP(predictor_sizes + [1], attention_size)

        # loss
        self.loss_func = nn.NLLLoss()


        self.to_common_device()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.load(params_path)

    def forward(self, state_graphs: List[Dict]):
        """
        :param state_graphs: [(t, graph)] * bs
        :param obj_info: (bs, obj_info_size)
        :return:
        """
        assert self._gnn_normalizers is not None

        state_len = [state_graph["nodes"].shape[0] for state_graph in state_graphs]
        state_mask = [torch.zeros(state_len_i, device=self.device) for state_len_i in state_len]
        state_mask = pad_sequence(state_mask, batch_first=True, padding_value=1.0)
        state_mask = state_mask.bool()                                          # (bs, t)

        state_embed = [self.graph_embedding_net(normalize_graph(traj, self._gnn_normalizers)) for traj in state_graphs] # [(t, attention_size)] * bs
        state_embed = pad_sequence(state_embed, batch_first=True)               # (bs, t, attention_size)
        state_feature = self.positional_encoding(state_embed)                   # (bs, t, attention_size)

        atten_pad_mask = [torch.zeros(state_len_i, device=self.device) for state_len_i in state_len]
        atten_pad_mask = pad_sequence(atten_pad_mask, batch_first=True, padding_value=1.0)
        atten_pad_mask = atten_pad_mask.bool()                                  # (bs, t)

        # (bs, t, attention_size)
        for l in self.atten_layers:
            state_feature = l(state_feature, key_padding_mask=atten_pad_mask)

        # (bs, t)
        pred_logits = self.imit_predictor(state_feature).squeeze(dim=-1)
        pred_logits[state_mask] = -np.inf

        return pred_logits

    def update(self, state_graphs: List[Dict], backjump_label: torch.Tensor, training=True):
        """
        :param state_graphs: [(t, graph)] * bs
        :param obj_info: (bs, obj_info_size)
        :param backjump_label: (bs,)
        :return:
        """
        # (bs, t)
        pred_logits = self.forward(state_graphs)

        loss = self.loss_func(F.log_softmax(pred_logits, dim=-1), backjump_label)
        loss_details = {"loss": loss}

        metrics = self.calculate_metrics(pred_logits, backjump_label)
        loss_details.update(metrics)

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss_details

    def compute_normalizers(self, state_graphs: List[Dict]):
        self._gnn_normalizers = compute_normalizers(state_graphs)

    def calculate_metrics(self, pred_logits, backjump_label):
        pred = pred_logits.argmax(dim=1)
        acc = (pred == backjump_label).float().mean()

        diff = pred - backjump_label

        le_mask = pred <= backjump_label
        le_perc = le_mask.float().mean()

        gt_mask = ~le_mask
        gt_perc = gt_mask.float().mean()

        metrics = {"acc": acc,
                   "le_percentage": le_perc,
                   "gt_perc": gt_perc}

        if le_mask.any():
            metrics["le_distance"] = -diff[le_mask]

        if gt_mask.any():
            metrics["gt_distance"] = diff[gt_mask]

        return metrics

    @torch.no_grad()
    def valid(self, state_graphs: List[Dict], backjump_label: torch.Tensor):
        return self.update(state_graphs, backjump_label, training=False)

    @torch.no_grad()
    def backjump(self, state_graphs: List[Dict], backjump_label: Optional[torch.Tensor] = None):
        pred_logits = self.forward(state_graphs)
        pred = pred_logits.argmax(dim=1).cpu().detach().numpy()

        if backjump_label is None:
            return pred

        metrics = self.calculate_metrics(pred_logits, backjump_label)

        return metrics

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    }, path)

    def load(self, path):
        if path is not None and os.path.exists(path):
            print("Imitation loaded", path)
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
