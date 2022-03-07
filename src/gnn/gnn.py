"""Graph neural network code.

A lot of this was copied and modified from code originally written by
Kelsey Allen.
"""
try:
    from .gnn_utils import *
except ImportError:
    from gnn_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def marshalling_func(graph):
    """Marshalling function rearranges objects and relations into interaction
    terms.

    :graph = the graph to be rearranged into interaction terms
    """

    receivers = torch.index_select(graph['nodes'], 0, graph['receivers'].view(-1))
    senders = torch.index_select(graph['nodes'], 0, graph['senders'].view(-1))
    if graph['globals'] is not None:
        global_idxs = np.repeat(np.arange(0, len(graph['globals'])), graph['n_edge'][:,0].cpu())
        global_tf = graph['globals'][global_idxs]
        return torch.cat((receivers, senders, graph['edges'], global_tf), dim=1), global_idxs
    return torch.cat((receivers, senders, graph['edges']), dim=1), []

def aggregation_func(graph):
    """Collects all effects that apply to each receiver object, merges them,
    and combines them with the object's attributes and externel effects to
    produce an input for the object model.

    :param graph: a graph to apply aggregation to
    """
    rec_m = prepare_receiver_matrix(graph)
    aggregated_effects = torch.mm(graph['edges'].t(), rec_m)
    if graph['globals']is not None:
        global_idxs = np.repeat(np.arange(0, len(graph['globals'])), graph['n_node'][:,0].cpu())
        global_tf = graph['globals'][global_idxs]
        return torch.cat((graph['nodes'], aggregated_effects.t(), global_tf), dim=1), global_idxs
    return torch.cat((graph['nodes'], aggregated_effects.t()), dim=1), []    

def prepare_receiver_matrix(graph):
    num_nodes = graph['nodes'].size()[0]
    num_edges = graph['edges'].size()[0]

    columns = torch.arange(0, num_nodes).long()
    if graph['receivers'].is_cuda:
        columns = columns.cuda()
    #idxs = torch.cat([columns.long().unsqueeze(0), graph['receivers'].data.unsqueeze(0)], dim=0)
    #rec_m = torch.sparse.FloatTensor(idxs, torch.ones([num_edges]), torch.Size([num_edges, num_nodes]))
    rec_m = graph['receivers'].view(-1)[:,None] == columns
    return rec_m.float()

def aggregate_globals(graph, global_node_idxs, global_edge_idxs):
    num_graphs = graph['globals'].size()[0]
    columns = torch.arange(0, num_graphs).long()

    node_idxs = torch.LongTensor(global_node_idxs)[:,None]
    edge_idxs = torch.LongTensor(global_edge_idxs)[:,None]

    if graph['globals'].is_cuda:
        columns = columns.cuda()
        node_idxs = node_idxs.cuda()
        edge_idxs = edge_idxs.cuda()

    nodes_agg = torch.mm(graph['nodes'].t(), (node_idxs == columns).float()).t()
    edges_agg = torch.mm(graph['edges'].t(), (edge_idxs == columns).float()).t()

    return torch.cat([graph['globals'], nodes_agg, edges_agg], dim=1)

def aggregate_globals_nodes(graph, global_node_idxs):
    num_graphs = graph['globals'].size()[0]
    columns = torch.arange(0, num_graphs).long()

    node_idxs = torch.LongTensor(global_node_idxs)[:,None]
    if graph['globals'].is_cuda:
        columns = columns.cuda()
        node_idxs = node_idxs.cuda()

    nodes_agg = torch.mm(graph['nodes'].t(), (node_idxs == columns).float()).t()

    return torch.cat([graph['globals'], nodes_agg], dim=1)

class GraphModel(nn.Module):
    def __init__(self, dims, poslinear=False,
                *args,
                **kwargs):
        super(GraphModel, self).__init__()
        self.poslinear = poslinear
        node_dim = dims[0]
        edge_dim = dims[1]
        global_dim = dims[2]
        self.params = []

        if 'node_encoder_layers' in kwargs.keys():
            self.node_encoder = MLP(kwargs['node_encoder_layers'], node_dim,
                poslinear=self.poslinear)
            node_dim = kwargs['node_encoder_layers'][-1]
            #self.add_module('node_encoder',self.node_encoder)
            self.params.append(self.node_encoder.parameters())
        if 'edge_encoder_layers' in kwargs.keys():
            self.edge_encoder = MLP(kwargs['edge_encoder_layers'], edge_dim,
                poslinear=self.poslinear)
            edge_dim = kwargs['edge_encoder_layers'][-1]
            #self.add_module('edge_encoder',self.edge_encoder)
            self.params.append(self.edge_encoder.parameters())
        if 'global_encoder_layers' in kwargs.keys():
            self.global_encoder = MLP(kwargs['global_encoder_layers'], global_dim,
                poslinear=self.poslinear)
            global_dim = kwargs['global_encoder_layers'][-1]
            #self.add_module('global_encoder',self.global_encoder)
            self.params.append(self.global_encoder.parameters())

        if 'node_model_layers' in kwargs.keys():
            input_dim = 2*node_dim + kwargs['edge_model_layers'][-1] + 2*global_dim
            self.node_model = MLP(kwargs['node_model_layers'], input_dim,
                poslinear=self.poslinear)
            #self.add_module('node_model',self.node_model)
            self.params.append(self.node_model.parameters())
        if 'edge_model_layers' in kwargs.keys():
            input_dim = 2*node_dim*2 + 2*edge_dim + 2*global_dim
            self.edge_model = MLP(kwargs['edge_model_layers'], input_dim,
                poslinear=self.poslinear)
            #self.add_module('edge_model',self.edge_model)
            self.params.append(self.edge_model.parameters())
        if 'global_model_layers' in kwargs.keys():
            input_dim = kwargs['node_model_layers'][-1] + kwargs['edge_model_layers'][-1] + 2*global_dim
            self.global_model = MLP(kwargs['global_model_layers'], input_dim,
                poslinear=self.poslinear)
            #self.add_module('global_model',self.global_model)
            self.params.append(self.global_model.parameters())

    def edges(self, graph):
        b, g = marshalling_func(graph)
        graph['edges'] = self.edge_model(b)
        return graph, g

    def nodes(self, graph):
        a, g = aggregation_func(graph)
        graph['nodes'] = self.node_model(a)
        return graph, g

    def globals(self, graph, global_edge_idxs, global_node_idxs):
        out = aggregate_globals(graph, global_node_idxs, global_edge_idxs)
        graph['globals'] = self.global_model(out)
        return graph

    def forward(self, graph):
        if graph['nodes'].is_cuda:
            graph['nodes'] = graph['nodes'].cuda()
            graph['edges'] = graph['edges'].cuda()
            graph['globals'] = graph['globals'].cuda()
            graph['receivers'] = graph['receivers'].cuda()
            graph['senders'] = graph['senders'].cuda()

        if hasattr(self, 'node_encoder'):
            graph['nodes'] = self.node_encoder(graph['nodes'])
        if hasattr(self, 'edge_encoder'):
            graph['edges'] = self.edge_encoder(graph['edges'])
        if hasattr(self, 'global_encoder'):
            graph['globals'] = self.global_encoder(graph['globals'])

        graph, eg = self.edges(graph)
        graph, ng = self.nodes(graph)
        if hasattr(self, 'global_encoder'):
            graph = self.globals(graph, eg, ng)

        return graph


class EncodeProcessDecode(GraphModel):
    def __init__(self, dims, num_steps, use_gpu, dropout=0., poslinear=False, **kwargs):
        super(EncodeProcessDecode, self).__init__(dims, poslinear=poslinear, **kwargs)
        if 'node_decoder_layers' in kwargs.keys():
            self.node_decoder = MLP(kwargs['node_decoder_layers'], kwargs['node_model_layers'][-1],
                dropout=dropout, poslinear=poslinear)
        if 'edge_decoder_layers' in kwargs.keys():
            self.edge_decoder = MLP(kwargs['edge_decoder_layers'], kwargs['edge_model_layers'][-1],
                dropout=dropout, poslinear=poslinear)
        if 'global_decoder_layers' in kwargs.keys():
            self.global_decoder = MLP(kwargs['global_decoder_layers'], kwargs['global_model_layers'][-1],
                dropout=dropout, poslinear=poslinear)
        self.num_steps = num_steps
        self.use_gpu = use_gpu
        self.dropout = dropout
        self.poslinear = poslinear

    def forward(self, graph):
        if graph['nodes'].is_cuda:
            assert self.use_gpu
            graph['nodes'] = graph['nodes'].cuda()
            graph['edges'] = graph['edges'].cuda()
            graph['globals'] = graph['globals'].cuda()
            graph['receivers'] = graph['receivers'].cuda()
            graph['senders'] = graph['senders'].cuda()

        if hasattr(self, 'node_encoder'):
            graph['nodes'] = self.node_encoder(graph['nodes'])
        if hasattr(self, 'edge_encoder'):
            graph['edges'] = self.edge_encoder(graph['edges'])
        if hasattr(self, 'global_encoder'):
            graph['globals'] = self.global_encoder(graph['globals'])

        output_graph = []
        latent0 = graph
        for steps in range(self.num_steps):
            graph = concat([latent0, graph], dim=1, use_gpu=self.use_gpu)
            graph, eg = self.edges(graph)
            graph, ng = self.nodes(graph)
            if hasattr(self, 'global_encoder'):
                graph = self.globals(graph, eg, ng)

            replacements = {
                'nodes': self.node_decoder(graph['nodes']),
                'edges':self.edge_decoder(graph['edges']),
            }

            if hasattr(self, 'global_decoder'):
               replacements['globals'] = self.global_decoder(graph['globals'])

            output_graph.append(replace(graph, replacements))

        return output_graph


# https://discuss.pytorch.org/t/positive-weights/19701/7
class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        return nn.functional.linear(input, self.weight.abs())

def MLP(layers, input_dim, dropout=0., poslinear=False):
    """Create MLP."""
    if poslinear:
        LinearLayer = PositiveLinear
    else:
        LinearLayer = nn.Linear
    mlp_layers = [LinearLayer(input_dim, layers[0])]

    for layer_num in range(0, len(layers)-1):
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(LinearLayer(layers[layer_num], layers[layer_num+1]))
    if not poslinear and len(layers) > 1:
        mlp_layers.append(nn.LayerNorm(mlp_layers[-1].weight.size()[:-1]))
        if dropout > 0:
            mlp_layers.append(nn.Dropout(p=dropout))

    return nn.Sequential(*mlp_layers)



def setup_dims(dataset, output_global_dim=None):
    """Extract dimes from dataset."""
    input_sample = dataset[0]['graph_input']
    output_sample = dataset[0]['graph_target']
    if isinstance(dataset[0]['graph_target'], list):
        # Recurrent: get dimensions from first target in sequence
        output_sample = output_sample[0]
    input_node_dim = input_sample['nodes'].shape[-1]
    input_edge_dim = input_sample['edges'].shape[-1]
    if input_sample['globals'] is not None:
        input_global_dim = input_sample['globals'].shape[-1]
    else:
        input_global_dim = 0
    output_node_dim = output_sample['nodes'].shape[-1]
    output_edge_dim = output_sample['edges'].shape[-1]
    if output_global_dim is None:
        if output_sample['globals'] is not None:
            output_global_dim = output_sample['globals'].shape[-1]
        else:
            output_global_dim = 0
    if isinstance(dataset[0]['graph_target'], list):
        # Recurrent: increase dimensions of globals to allow for hidden state
        input_global_dim += RECURRENT_HIDDEN_STATE_SIZE
        output_global_dim += RECURRENT_HIDDEN_STATE_SIZE
    return [input_node_dim, input_edge_dim, input_global_dim,
            output_node_dim, output_edge_dim, output_global_dim]

def setup_graph_net(graph_dataset, use_gpu=False, num_steps=1, layer_size=16, output_dim=1,
                    dropout=0., output_global_dim=None, poslinear=False):
    """Create an EncodeProcessDecode graphnet using the dimensions found in the
    dataset."""
    dims = setup_dims(graph_dataset, output_global_dim=output_global_dim)
    include_globals = dims[-1] > 0

    enc_dims = [layer_size, layer_size]
    enc_layers = {'nodes':[layer_size, enc_dims[0]],'edges':[layer_size,enc_dims[1]]}
    if include_globals:
        enc_layers['globals'] =[layer_size,enc_dims[1]]

    in_layers = {'nodes':[layer_size, layer_size], 'edges':[layer_size, layer_size]}
    if include_globals:
        in_layers['globals'] = [layer_size,layer_size]

    dec_layers = {'nodes':[dims[3]], 'edges' : [dims[4]]}
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

    encprocdec = EncodeProcessDecode(dims, num_steps, use_gpu, **layer_dict,
                                     dropout=dropout, poslinear=poslinear)

    return encprocdec
