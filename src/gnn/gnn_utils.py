from predicators.src.gnn.gnn_dataset import create_super_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import time
import copy
import matplotlib.pyplot as plt
import os

RECURRENT_HIDDEN_STATE_SIZE = 32


def train_model(model, dataloaders, criterion, optimizer, use_gpu, print_iter=10, 
                save_iter=100, save_folder='/tmp', num_epochs=1000, global_criterion=None,
                return_last_model_weights=False, target_nodes_only=False, do_validation=True):
    """Optimize the model and save checkpoints."""
    since = time.time()

    best_seen_model_weights = None # as measured on validation loss
    best_seen_model_train_loss = np.inf
    best_seen_running_validation_loss = np.inf

    if use_gpu:
        model = model.cuda()
        if criterion is not None:
            criterion = criterion.cuda()
        if global_criterion is not None:
            global_criterion = global_criterion.cuda()

    for epoch in range(num_epochs):
        if epoch % print_iter == 0:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1), flush=True)
            print('-' * 10, flush=True)
        # Each epoch has a training and validation phase
        if epoch % print_iter == 0 and do_validation:
            phases = ['train','val']
        else:
            phases = ['train']
        
        running_loss = {'train':0.0,'val':0.0}
        if not do_validation:
            del running_loss['val']

        for phase in phases:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            # Iterate over data.
            for data in dataloaders[phase]:
                inputs = data['graph_input']
                targets = data['graph_target']
                
                if use_gpu:
                    for key in inputs.keys():
                        inputs[key] = inputs[key].cuda()
                for key in targets.keys():
                    if use_gpu:
                        targets[key] = targets[key].cuda()
                    if targets[key] is not None:
                        targets[key] = targets[key].detach()

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs.copy())
                output = outputs[-1]

                loss = 0.
                if criterion is not None:
                    loss += criterion(output['nodes'], targets['nodes'])
                    if not target_nodes_only and targets['edges'].shape[1] > 0:
                        loss += criterion(output['edges'], targets['edges'])

                if global_criterion is not None:
                    if isinstance(global_criterion, nn.CrossEntropyLoss):
                        global_loss = global_criterion(
                            output['globals'],
                            targets['globals'].long().view(-1)) # assumes crossentropyloss
                    else:
                        global_loss = global_criterion(
                            output['globals'],
                            targets['globals'])
                    loss += global_loss

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss[phase] += loss.item()

        if epoch % print_iter == 0:
            print("running_loss:", running_loss, flush=True)

        if epoch % save_iter == 0:
            # save_path = os.path.join(save_folder, "model" + str(epoch) + ".pt")
            # torch.save(model.state_dict(), save_path)
            # print("Saved model checkpoint {}".format(save_path))

            if do_validation and running_loss['val'] < best_seen_running_validation_loss:
                best_seen_running_validation_loss = running_loss['val']
                best_seen_model_weights = model.state_dict()
                best_seen_model_train_loss = running_loss['train']
                print("Found new best model with validation loss {} at epoch {}".format(
                    best_seen_running_validation_loss, epoch), flush=True)

    time_elapsed = time.time() - since

    if return_last_model_weights or not do_validation:
        print('Training complete in {:.0f}m {:.0f}s with train loss {:.5f}'.format(time_elapsed // 60, time_elapsed % 60, running_loss['train']), flush=True)
        return model.state_dict()
    print('Training complete in {:.0f}m {:.0f}s with train loss {:.5f} and validation loss {:.5f}'.format(time_elapsed // 60, time_elapsed % 60, best_seen_model_train_loss, best_seen_running_validation_loss), flush=True)

    return best_seen_model_weights

def train_model_recurrent(
        model, graph_dataset, criterion, optimizer, print_iter=10,
        save_iter=100, save_folder='/tmp', num_epochs=1000, global_criterion=None,
        target_nodes_only=False):
    """Optimize the recurrent model and save checkpoints.

    Target data is expected to be a sequence of graphs rather than a
    single graph. Hidden state is automatically added into the globals.
    """
    since = time.time()

    for epoch in range(num_epochs):
        if epoch % print_iter == 0:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1), flush=True)
            print('-' * 10, flush=True)
        running_loss = {'train':0.0}
        model.train(True)  # Set model to training mode
        # Iterate over data.
        for datapoint in graph_dataset:
            inp = datapoint["graph_input"]
            target_sequence = datapoint["graph_target"]
            assert isinstance(target_sequence, list)
            # zero the parameter gradients
            optimizer.zero_grad()
            # initialize hidden state
            hidden_state = torch.from_numpy(np.random.normal(
                scale=0.001, size=(1, RECURRENT_HIDDEN_STATE_SIZE)).astype(np.float32))
            # compute loss over sequence
            loss = 0.
            for target in target_sequence:
                inps = create_super_graph([inp])
                inps["globals"] = torch.cat([inps["globals"], hidden_state], dim=1)
                output = model(inps.copy())[-1]
                targets = create_super_graph([target])
                if criterion is not None:
                    loss += criterion(output['nodes'], targets['nodes'])
                    if not target_nodes_only and targets['edges'].shape[1] > 0:
                        loss += criterion(output['edges'], targets['edges'])
                if global_criterion is not None:
                    global_loss = global_criterion(output['globals'], targets['globals'])
                    loss += global_loss
                assert output["globals"].ndim == 2 and output["globals"].shape[0] == 1
                # update hidden state
                if RECURRENT_HIDDEN_STATE_SIZE > 0:
                    hidden_state = output["globals"][:, -RECURRENT_HIDDEN_STATE_SIZE:]

            # optimize
            loss.backward()
            optimizer.step()

            running_loss["train"] += loss.item()

        if epoch % print_iter == 0:
            print("running_loss:", running_loss, flush=True)

        if epoch % save_iter == 0:
            save_path = os.path.join(save_folder, "model" + str(epoch) + ".pt")
            torch.save(model.state_dict(), save_path)
            print("Saved model checkpoint {}".format(save_path))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), flush=True)

    return model.state_dict()


def compute_normalizers(data):
    node_data, edge_data, global_data = [], [], []
    for graph in data:
        node_data.extend(graph["nodes"])
        edge_data.extend(graph["edges"])
        global_data.append(graph["globals"])
    node_data = np.array(node_data)
    edge_data = np.array(edge_data)
    global_data = np.array(global_data)
    node_normalizers = _compute_normalizer_array(node_data)
    edge_normalizers = _compute_normalizer_array(edge_data)
    global_normalizers = _compute_normalizer_array(global_data)
    return {"nodes": node_normalizers, "edges": edge_normalizers,
            "globals": global_normalizers}


def _compute_normalizer_array(array_data):
    # Handle an edge case.
    if len(array_data) == 0:
        return 0, 1
    shift = np.min(array_data, axis=0)
    scale = np.max(array_data - shift, axis=0)
    shift[np.where(scale == 0)] = 0
    scale[np.where(scale == 0)] = 1
    return shift, scale


def normalize_graph(graph, normalizer, invert=False):
    assert set(normalizer.keys()).issubset(set(graph.keys()))
    if invert:
        transform = _invert_normalize_array
    else:
        transform = _normalize_array
    new_graph = {}
    for k in graph:
        if k in normalizer:
            new_graph[k] = transform(graph[k], normalizer[k])
        else:
            assert k in ['n_node', 'n_edge', 'senders', 'receivers']
            new_graph[k] = graph[k]
    return new_graph


def _normalize_array(array_data, normalizer):
    shift, scale = normalizer
    return (array_data - shift) / scale


def _invert_normalize_array(array_data, normalizer):
    shift, scale = normalizer
    return (array_data * scale) + shift


def get_model_predictions(model, dataloader, use_gpu=False):
    predictions = []
    model.train(False)
    model.eval()
    for data in dataloader:
        inputs = data['graph_input']
        
        if use_gpu:
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()

        outputs = model(inputs.copy())
        graphs = split_graphs(convert_to_data(outputs[-1]), use_gpu=use_gpu)
        for graph in graphs:
            graph['nodes'] = graph['nodes'].numpy()
            graph['senders'] = graph['senders'].numpy()
            graph['receivers'] = graph['receivers'].numpy()
            graph['edges'] = graph['edges'].numpy()
            if graph['globals'] is not None:
                graph['globals'] = graph['globals'].numpy()
            graph['n_node'] = graph['n_node'].item()
            graph['n_edge'] = graph['n_edge'].item()
            predictions.append(graph)
    return predictions

def get_single_model_prediction(model, single_input, use_gpu=False):
    model.train(False)
    model.eval()
    inputs = create_super_graph([single_input])
    if use_gpu:
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda()
    outputs = model(inputs.copy())
    graphs = split_graphs(convert_to_data(outputs[-1]), use_gpu=use_gpu)
    assert len(graphs) == 1
    graph = graphs[0]
    graph['nodes'] = graph['nodes'].numpy()
    graph['senders'] = graph['senders'].numpy()
    graph['receivers'] = graph['receivers'].numpy()
    graph['edges'] = graph['edges'].numpy()
    if graph['globals'] is not None:
        graph['globals'] = graph['globals'].numpy()
    graph['n_node'] = graph['n_node'].item()
    graph['n_edge'] = graph['n_edge'].item()
    return graph

def get_single_model_prediction_recurrent(model, single_input, hidden_state):
    model.train(False)
    model.eval()
    hidden_state = torch.from_numpy(hidden_state)
    inputs = create_super_graph([single_input])
    inputs["globals"] = torch.cat([inputs["globals"], hidden_state], dim=1)
    outputs = model(inputs.copy())
    graphs = split_graphs(convert_to_data(outputs[-1]), use_gpu=False)
    assert len(graphs) == 1
    graph = graphs[0]
    graph['nodes'] = graph['nodes'].numpy()
    graph['senders'] = graph['senders'].numpy()
    graph['receivers'] = graph['receivers'].numpy()
    graph['edges'] = graph['edges'].numpy()
    if graph['globals'] is not None:
        graph['globals'] = graph['globals'].numpy()
    graph['n_node'] = graph['n_node'].item()
    graph['n_edge'] = graph['n_edge'].item()
    assert graph['globals'].ndim == 1
    if RECURRENT_HIDDEN_STATE_SIZE > 0:
        graph['globals'], next_hidden_state = (
            graph['globals'][:-RECURRENT_HIDDEN_STATE_SIZE],
            graph['globals'][-RECURRENT_HIDDEN_STATE_SIZE:])
    else:
        next_hidden_state = np.array([], dtype=np.float32)
    return graph, np.expand_dims(next_hidden_state, axis=0)

def get_multi_model_predictions(model, inputs, use_gpu=False):
    model.train(False)
    model.eval()
    inputs = create_super_graph(inputs)
    if use_gpu:
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda()
    outputs = model(inputs.copy())
    graphs = split_graphs(convert_to_data(outputs[-1]), use_gpu=use_gpu)
    out = []
    for graph in graphs:
        graph['nodes'] = graph['nodes'].numpy()
        graph['senders'] = graph['senders'].numpy()
        graph['receivers'] = graph['receivers'].numpy()
        graph['edges'] = graph['edges'].numpy()
        if graph['globals'] is not None:
            graph['globals'] = graph['globals'].numpy()
        graph['n_node'] = graph['n_node'].item()
        graph['n_edge'] = graph['n_edge'].item()
        out.append(graph)
    return out

def _compute_stacked_offsets(sizes, repeats, numpy=False, use_gpu=True):
  """Computes offsets to add to indices of stacked np arrays.

  When a set of np arrays are stacked, the indices of those from the second on
  must be offset in order to be able to index into the stacked np array. This
  computes those offsets.
  Args:
    sizes: A 1D sequence of np arrays of the sizes per graph.
    repeats: A 1D sequence of np arrays of the number of repeats per graph.
  Returns:
    The index offset per graph.
  """
  idxs = np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)
  if numpy:
    return idxs
  else:
    if use_gpu:
        return torch.LongTensor(idxs).cuda()
    return torch.LongTensor(idxs)

def convert_to_data(graph):
    for key in graph.keys():
        if graph[key] is not None:
            graph[key] = graph[key].data
    return graph

def replace(graph, graph_dict):
  out_graph = graph.copy()
  for key in graph_dict.keys():
    out_graph[key] = graph_dict[key]
  return out_graph

def _unstack(array):
  """Similar to `tf.unstack`."""
  num_splits = int(array.shape[0])
  return [torch.squeeze(x, dim=0) for x in np.split(array, num_splits, axis=0)]

def split_graphs(graph, use_gpu=True):
  """Splits the stored data into a list of individual data dicts.

  Each list is a dictionary with fields NODES, EDGES, GLOBALS, RECEIVERS,
  SENDERS.
  Args:
    graph: A `graphs.GraphsTuple` instance containing numpy arrays.
  Returns:
    A list of the graph data dictionaries. The GLOBALS field is a tensor of
      rank at least 1, as the RECEIVERS and SENDERS field (which have integer
      values). The NODES and EDGES fields have rank at least 2.
  """
  offset = _compute_stacked_offsets(graph['n_node'].view(-1), graph['n_edge'].view(-1), numpy=False, use_gpu=use_gpu)
  nodes_splits = np.cumsum(graph['n_node'][:-1])
  edges_splits = np.cumsum(graph['n_edge'][:-1])
  graph_of_lists = collections.defaultdict(lambda: [])
  if graph['nodes'] is not None:
    graph_of_lists['nodes'] = np.split(graph['nodes'], nodes_splits)
  if graph['edges'] is not None:
    graph_of_lists['edges'] = np.split(graph['edges'], edges_splits)
  if graph['receivers'] is not None:
    graph_of_lists['receivers'] = np.split(graph['receivers'] - offset, edges_splits)
    graph_of_lists['senders'] = np.split(graph['senders'] - offset, edges_splits)
  if graph['globals'] is not None:
    graph_of_lists['globals'] = _unstack(graph['globals'])

  n_graphs = graph['n_node'].shape[0]
  # Make all fields the same length.
  for k in ['nodes','edges','globals']:
    graph_of_lists[k] += [None] * (n_graphs - len(graph_of_lists[k]))
  graph_of_lists['n_node'] = graph['n_node']
  graph_of_lists['n_edge'] = graph['n_edge']

  result = []
  for index in range(n_graphs):
    result.append({field: graph_of_lists[field][index] for field in ['nodes','edges','receivers','senders','globals','n_node','n_edge']})
  return result

def concat(input_graphs, dim, use_gpu=True):
  """In all cases, the NODES, EDGES and GLOBALS dimension are concatenated
  along `dim` (if a fields is `None`, the concatenation is just a `None`).

  If `dim` == 0, then the graphs are concatenated along the (underlying) batch
  dimension, i.e. the RECEIVERS, SENDERS, N_NODE and N_EDGE fields of the tuples
  are also concatenated together.
  If `dim` != 0, then there is an underlying asumption that the receivers,
  SENDERS, N_NODE and N_EDGE fields of the graphs in `values` should all match,
  but this is not checked by this op.
  The graphs in `input_graphs` should have the same set of keys for which the
  corresponding fields is not `None`.
  Args:
    input_graphs: A list of `graphs.GraphsTuple` objects containing `Tensor`s
      and satisfying the constraints outlined above.
    dim: An dim to concatenate on.
    name: (string, optional) A name for the operation.
  Returns: An op that returns the concatenated graphs.
  Raises:
    ValueError: If `values` is an empty list, or if the fields which are `None`
      in `input_graphs` are not the same for all the graphs.
  """
  if not input_graphs:
    raise ValueError("List argument `input_graphs` is empty")
  if len(input_graphs) == 1:
    return input_graphs[0]
  nodes = [gr['nodes'] for gr in input_graphs if gr['nodes'] is not None]
  edges = [gr['edges'] for gr in input_graphs if gr['edges'] is not None]
  globals_ = [gr['globals'] for gr in input_graphs if gr['globals'] is not None]

  nodes = torch.cat(nodes, dim) if nodes else None
  edges = torch.cat(edges, dim) if edges else None
  if globals_:
    globals_ = torch.cat(globals_, dim)
  else:
    globals_ = None

  output = replace(input_graphs[0],{'nodes':nodes, 'edges':edges, 'globals':globals_})
  if dim != 0:
    return output

  test = [torch.sum(gr['n_node']) for gr in input_graphs]
  n_node_per_tuple = torch.stack(
      [torch.sum(gr['n_node']) for gr in input_graphs])
  n_edge_per_tuple = torch.stack(
      [torch.sum(gr['n_edge']) for gr in input_graphs])
  offsets = _compute_stacked_offsets(n_node_per_tuple, n_edge_per_tuple, use_gpu=use_gpu)
  n_node = torch.cat(
      [gr['n_node'] for gr in input_graphs], dim=0)
  n_edge = torch.cat(
      [gr['n_edge'] for gr in input_graphs], dim=0)
  receivers = [
      gr['receivers'] for gr in input_graphs if gr['receivers'] is not None
  ]
  receivers = receivers or None
  if receivers:
    receivers = torch.cat(receivers, dim) + offsets
  senders = [gr['senders'] for gr in input_graphs if gr['senders'] is not None]
  senders = senders or None
  if senders:
    senders = torch.cat(senders, dim) + offsets
  return replace(output, {'receivers':receivers, 'senders':senders, 'n_node':n_node, 'n_edge':n_edge})
