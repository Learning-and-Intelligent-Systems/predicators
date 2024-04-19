"""Graph neural network core code.

A lot of this was copied and modified from code originally written by
Kelsey Allen.
"""

import abc
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from predicators.gnn.gnn_utils import GraphDictDataset, concat_graphs, \
    replace_graph
from predicators.ml_models import DeviceTrackingModule, _get_torch_device
from predicators.structs import Array


def _marshalling_func(graph: Dict) -> Tuple[torch.Tensor, Array]:
    """Marshalling function rearranges objects and relations into interaction
    terms."""

    receivers = torch.index_select(graph['nodes'], 0,
                                   graph['receivers'].view(-1))
    senders = torch.index_select(graph['nodes'], 0, graph['senders'].view(-1))
    global_idxs = np.repeat(np.arange(0, len(graph['globals'])),
                            graph['n_edge'][:, 0].cpu())
    global_tf = graph['globals'][global_idxs]
    return torch.cat((receivers, senders, graph['edges'], global_tf),
                     dim=1), global_idxs


def _aggregation_func(graph: Dict) -> Tuple[torch.Tensor, Array]:
    """Collects all effects that apply to each receiver object, merges them,
    and combines them with the object's attributes and externel effects to
    produce an input for the object model."""
    rec_m = _prepare_receiver_matrix(graph)
    aggregated_effects = torch.mm(graph['edges'].t(), rec_m)
    global_idxs = np.repeat(np.arange(0, len(graph['globals'])),
                            graph['n_node'][:, 0].cpu())
    global_tf = graph['globals'][global_idxs]
    return torch.cat((graph['nodes'], aggregated_effects.t(), global_tf),
                     dim=1), global_idxs


def _prepare_receiver_matrix(graph: Dict) -> torch.Tensor:
    num_nodes = graph['nodes'].size()[0]
    columns = torch.arange(0, num_nodes).long().to(graph['nodes'].device)
    rec_m = graph['receivers'].view(-1)[:, None] == columns
    return rec_m.float()


def _aggregate_globals(graph: Dict, global_node_idxs: Array,
                       global_edge_idxs: Array) -> torch.Tensor:
    num_graphs = graph['globals'].size()[0]
    device = graph['globals'].device
    columns = torch.arange(0, num_graphs).long().to(device)

    node_idxs = torch.LongTensor(global_node_idxs)[:, None].to(device)
    edge_idxs = torch.LongTensor(global_edge_idxs)[:, None].to(device)

    nodes_agg = torch.mm(graph['nodes'].t(),
                         (node_idxs == columns).float()).t()
    edges_agg = torch.mm(graph['edges'].t(),
                         (edge_idxs == columns).float()).t()

    return torch.cat([graph['globals'], nodes_agg, edges_agg], dim=1)


class GraphModel(DeviceTrackingModule, abc.ABC):
    """General GNN architecture."""

    def __init__(self, dims: List[int], use_torch_gpu: bool, **kwargs: List[int]) -> None:
        super().__init__(_get_torch_device(use_torch_gpu))  # type: ignore
        node_dim = dims[0]
        edge_dim = dims[1]
        global_dim = dims[2]
        self.params = []

        if 'node_encoder_layers' in kwargs:
            self.node_encoder = MLP(kwargs['node_encoder_layers'], node_dim)
            node_dim = kwargs['node_encoder_layers'][-1]
            self.params.append(self.node_encoder.parameters())
        if 'edge_encoder_layers' in kwargs:
            self.edge_encoder = MLP(kwargs['edge_encoder_layers'], edge_dim)
            edge_dim = kwargs['edge_encoder_layers'][-1]
            self.params.append(self.edge_encoder.parameters())
        if 'global_encoder_layers' in kwargs:
            self.global_encoder = MLP(kwargs['global_encoder_layers'],
                                      global_dim)
            global_dim = kwargs['global_encoder_layers'][-1]
            self.params.append(self.global_encoder.parameters())

        if 'node_model_layers' in kwargs:
            input_dim = (2 * node_dim + kwargs['edge_model_layers'][-1] +
                         2 * global_dim)
            self.node_model = MLP(kwargs['node_model_layers'], input_dim)
            self.params.append(self.node_model.parameters())
        if 'edge_model_layers' in kwargs:
            input_dim = 2 * node_dim * 2 + 2 * edge_dim + 2 * global_dim
            self.edge_model = MLP(kwargs['edge_model_layers'], input_dim)
            self.params.append(self.edge_model.parameters())
        if 'global_model_layers' in kwargs:
            input_dim = (kwargs['node_model_layers'][-1] +
                         kwargs['edge_model_layers'][-1] + 2 * global_dim)
            self.global_model = MLP(kwargs['global_model_layers'], input_dim)
            self.params.append(self.global_model.parameters())
        self.to_common_device()

    def edges(self, graph: Dict) -> Tuple[Dict, Array]:
        """Run marshalling function."""
        b, g = _marshalling_func(graph)
        graph['edges'] = self.edge_model(b)
        return graph, g

    def nodes(self, graph: Dict) -> Tuple[Dict, Array]:
        """Run aggregation function."""
        a, g = _aggregation_func(graph)
        graph['nodes'] = self.node_model(a)
        return graph, g

    def globals(self, graph: Dict, global_edge_idxs: Array,
                global_node_idxs: Array) -> Dict:
        """Aggregate globals."""
        out = _aggregate_globals(graph, global_node_idxs, global_edge_idxs)
        graph['globals'] = self.global_model(out)
        return graph


class EncodeProcessDecode(GraphModel):
    """Encode-process-decode GNN architecture."""

    def __init__(self, dims: List[int], num_steps: int, use_torch_gpu: bool,
                 **kwargs: List[int]) -> None:
        super().__init__(dims, use_torch_gpu, **kwargs)
        if 'node_decoder_layers' in kwargs:
            self.node_decoder = MLP(kwargs['node_decoder_layers'],
                                    kwargs['node_model_layers'][-1])
        if 'edge_decoder_layers' in kwargs:
            self.edge_decoder = MLP(kwargs['edge_decoder_layers'],
                                    kwargs['edge_model_layers'][-1])
        if 'global_decoder_layers' in kwargs:
            self.global_decoder = MLP(kwargs['global_decoder_layers'],
                                      kwargs['global_model_layers'][-1])
        self.num_steps = num_steps
        self.to_common_device()

    def forward(self, graph: Dict) -> List[Dict]:
        """Torch forward model."""
        if hasattr(self, 'node_encoder'):
            graph['nodes'] = self.node_encoder(graph['nodes'])
        if hasattr(self, 'edge_encoder'):
            graph['edges'] = self.edge_encoder(graph['edges'])
        if hasattr(self, 'global_encoder'):
            graph['globals'] = self.global_encoder(graph['globals'])

        output_graph = []
        latent0 = graph
        for _ in range(self.num_steps):
            graph = concat_graphs([latent0, graph], dim=1)
            graph, eg = self.edges(graph)
            graph, ng = self.nodes(graph)
            if hasattr(self, 'global_encoder'):
                graph = self.globals(graph, eg, ng)

            replacements = {
                'nodes': self.node_decoder(graph['nodes']),
                'edges': self.edge_decoder(graph['edges']),
            }

            if hasattr(self, 'global_decoder'):
                replacements['globals'] = self.global_decoder(graph['globals'])

            output_graph.append(replace_graph(graph, replacements))

        return output_graph


def MLP(layers: List[int], input_dim: int) -> nn.Sequential:
    """Create MLP."""
    LinearLayer = nn.Linear
    if not input_dim:
        mlp_layers: List[nn.Module] = [LinearLayer(input_dim, layers[0], bias=False), LinearLayer(layers[0], layers[0])]
    else:
        mlp_layers: List[nn.Module] = [LinearLayer(input_dim, layers[0])]

    for layer_num in range(0, len(layers) - 1):
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(LinearLayer(layers[layer_num],
                                      layers[layer_num + 1]))
    if len(layers) > 1:
        mlp_layers.append(nn.LayerNorm(
            mlp_layers[-1].weight.size()[:-1]))  # type: ignore

    return nn.Sequential(*mlp_layers)


def _setup_dims(dataset: GraphDictDataset) -> List[int]:
    """Extract dimensions from dataset."""
    input_sample = dataset[0]['graph_input']
    output_sample = dataset[0]['graph_target']
    input_node_dim = input_sample['nodes'].shape[-1]
    input_edge_dim = input_sample['edges'].shape[-1]
    assert input_sample['globals'] is not None
    input_global_dim = input_sample['globals'].shape[-1]
    output_node_dim = output_sample['nodes'].shape[-1]
    output_edge_dim = output_sample['edges'].shape[-1]
    assert output_sample['globals'] is not None
    output_global_dim = output_sample['globals'].shape[-1]
    return [
        input_node_dim, input_edge_dim, input_global_dim, output_node_dim,
        output_edge_dim, output_global_dim
    ]


def setup_graph_net(graph_dataset: GraphDictDataset, num_steps: int,
                    layer_size: int, use_torch_gpu: bool) -> EncodeProcessDecode:
    """Create an EncodeProcessDecode GNN using the dimensions found in the
    dataset."""
    dims = _setup_dims(graph_dataset)
    include_globals = dims[-1] > 0

    enc_dims = [layer_size, layer_size]
    enc_layers = {
        'nodes': [layer_size, enc_dims[0]],
        'edges': [layer_size, enc_dims[1]]
    }
    if include_globals:
        enc_layers['globals'] = [layer_size, enc_dims[1]]

    in_layers = {
        'nodes': [layer_size, layer_size],
        'edges': [layer_size, layer_size]
    }
    if include_globals:
        in_layers['globals'] = [layer_size, layer_size]

    dec_layers = {'nodes': [dims[3]], 'edges': [dims[4]]}
    if include_globals:
        dec_layers['globals'] = [dims[-1]]
    layer_dict = {}

    layer_dict['node_encoder_layers'] = enc_layers['nodes']
    layer_dict['edge_encoder_layers'] = enc_layers['edges']
    if include_globals:
        layer_dict['global_encoder_layers'] = enc_layers['globals']

    layer_dict['node_model_layers'] = in_layers['nodes']
    layer_dict['edge_model_layers'] = in_layers['edges']
    if include_globals:
        layer_dict['global_model_layers'] = in_layers['globals']

    layer_dict['node_decoder_layers'] = dec_layers['nodes']
    layer_dict['edge_decoder_layers'] = dec_layers['edges']
    if include_globals:
        layer_dict['global_decoder_layers'] = dec_layers['globals']

    encprocdec = EncodeProcessDecode(dims, num_steps, use_torch_gpu, **layer_dict)

    return encprocdec
