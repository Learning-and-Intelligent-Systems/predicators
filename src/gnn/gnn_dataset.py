from __future__ import print_function, division
from torch.utils.data import Dataset
import os
import torch
import pandas as pd
import numpy as np


class GraphDictDataset(Dataset):
    def __init__(self, graph_dicts_input, graph_dicts_target):
        self.graph_dicts_input = graph_dicts_input
        self.graph_dicts_target = graph_dicts_target

    def __len__(self):
        return len(self.graph_dicts_input)

    def __getitem__(self, idx):
        sample = {'graph_input':self.graph_dicts_input[idx], 'graph_target':self.graph_dicts_target[idx]}
        return sample


def create_super_graph(batches):
    nodes = batches[0]['nodes']
    edges = batches[0]['edges']
    receivers = batches[0]['receivers'][:,None]
    senders = batches[0]['senders'][:,None]
    globals = batches[0]['globals']
    if globals is not None and len(globals.shape) < 2:
        globals = globals[None,:]

    num_nodes = np.array(batches[0]['n_node'],ndmin=2)
    num_edges = np.array(batches[0]['n_edge'],ndmin=2)

    for i, b in enumerate(batches[1:]):
        nodes = np.vstack((nodes, b['nodes']))
        receivers = np.vstack((receivers, (b['receivers'] + np.sum(num_nodes))[:,None]))
        senders = np.vstack((senders, (b['senders'] + np.sum(num_nodes))[:,None]))
        if globals is not None:
            globals = np.vstack((globals, b['globals']))
        edges = np.vstack((edges, b['edges']))

        num_nodes = np.vstack((num_nodes, b['n_node']))
        num_edges = np.vstack((num_edges, b['n_edge']))

    return {
        'n_node': torch.from_numpy(num_nodes), 
        'n_edge': torch.from_numpy(num_edges), 
        'nodes': torch.from_numpy(nodes).float().requires_grad_(), 
        'edges': torch.from_numpy(edges).float().requires_grad_(), 
        'receivers': torch.LongTensor(list(map(int, receivers))), 
        'senders': torch.LongTensor(list(map(int, senders))),
        'globals': torch.from_numpy(globals).float().requires_grad_() if globals is not None else None,
    }


def graph_batch_collate(batch):
    """Assumes batch is a dictionary where each key contains a list of
    graphs."""
    error_msg = "batch must contain dict of graphs; found {}"
    return {key: create_super_graph([d[key] for d in batch]) for key in batch[0]}




