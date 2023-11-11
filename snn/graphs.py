from collections import OrderedDict

import numpy as np
from numba.core.types import int32
from utils import *


@jitclass(OrderedDict([
    ('out_edges', numbaListType(int32[::1])),
    ('in_edges', numbaListType(int32[::1])),
    ('spikes', int32[::1]),
]))
class DirectedEdgeListGraph:

    def __init__(self):
        self.spikes = njit_empty_array()
        self.in_edges = numbaList(njit_empty_list())
        self.out_edges = numbaList(njit_empty_list())

    def add_node(self, node):
        node._id = len(self.out_edges)
        self.spikes = np.append(self.spikes, 0).astype(np.int32)
        self.in_edges.append(njit_empty_array())
        self.out_edges.append(njit_empty_array())

    def add_graph(self, graph):
        new_id_offset = len(self.out_edges)
        self.spikes = np.concatenate((self.spikes, graph.spikes))
        for edge in range(len(graph.in_edges)):
            graph.in_edges[edge] += new_id_offset
            self.in_edges.append(graph.in_edges[edge])
        # self.in_edges = self.in_edges + graph.in_edges
        for edge in range(len(graph.out_edges)):
            graph.out_edges[edge] += new_id_offset
            self.out_edges.append(graph.out_edges[edge])
        # self.out_edges = self.out_edges + graph.out_edges
        return new_id_offset

    def connect(self, source_node, target):
        self.connect_by_id(source_node._id, target._id)

    def connect_by_id(self, source_id, target_id):
        source_id = np.int32(source_id)
        target_id = np.int32(target_id)
        self.in_edges[target_id] = np.append(self.in_edges[target_id], source_id).astype(np.int32)
        self.out_edges[source_id] = np.append(self.out_edges[source_id], target_id).astype(np.int32)

    def update_spike(self, neuron, spike):
        self.spikes[neuron._id] = spike

    def get_input_spikes_to(self, neuron):
        input_neurons = self.in_edges[neuron._id]
        if len(input_neurons) > 0:
            return self.spikes[input_neurons]
        return njit_empty_array()

    def size(self):
        return len(self.out_edges)

    def remove_any_connections(self, to_remove):
        for nid in to_remove:
            for target in self.out_edges[nid]:
                self.in_edges[target] = np.delete(
                    self.in_edges[target],
                    np.where(self.in_edges[target] == nid)[0].astype(np.int32)
                )

            for source in self.in_edges[nid]:
                self.out_edges[source] = np.delete(
                    self.out_edges[source],
                    np.where(self.out_edges[source] == nid)[0].astype(np.int32)
                )
            self.in_edges[nid] = njit_empty_array()
            self.out_edges[nid] = njit_empty_array()
            self.spikes[nid] = 0


@njit
def njit_empty_list():
    return [np.array([np.int32(0)]) for _ in range(0)]


@njit
def njit_empty_array():
    return np.array([np.int32(0) for _ in range(0)])



def plot_network(network):
    G = nx.DiGraph()
    pos = {}
    nodes_color = {}
    nodes_size = {}
    nodes_labels = {}

    input_size = len(network.layers_neurons[0].neurons[0].synapses_weights)
    column_length = max(len(layer.neurons) for layer in network.layers_neurons)
    column_length = max(column_length, input_size)/2
    rows_length = len(network.layers_neurons) + 2   # +2 for input and output
    plt.figure(figsize=(rows_length, 1 + (column_length // 1)), dpi=120)

    for i, layer in enumerate(network.layers_neurons):
        for j, neuron in enumerate(layer.neurons):
            gap = column_length/len(layer.neurons)
            nodes_labels[neuron._id] = neuron.label or neuron._id
            pos[neuron._id] = [i+.5, j * gap + gap/2]
            nodes_color[neuron._id] = 'blue'
            nodes_size[neuron._id] = 100

    # Create nodes for "input" layer
    for i in range(1, input_size + 1):
        nodes_color[-i] = 'black'
        nodes_size[-i] = 12
        gap = column_length / input_size
        pos[-i] = [0, (i - 1) * gap + gap / 2]

        # Connect the input to first layer
        for j, neuron in enumerate(network.layers_neurons[0].neurons):
            label = neuron.label or neuron._id
            G.add_edge(-i, label, color='black')

    # Create nodes for "output" layer
    for i, neuron in enumerate(network.layers_neurons[-1].neurons):
        target = network.neurons_count + i
        nodes_color[target] = 'black'
        nodes_size[target] = 12
        gap = column_length / len(network.layers_neurons[-1].neurons)
        pos[target] = [len(network.layers_neurons) + 1, i * gap + gap / 2]
        label = neuron.label or neuron._id
        G.add_edge(label, target, color='black')

    for out_edge, in_edges in enumerate(network.spikes_graph.out_edges):
        for in_edge in in_edges:
            G.add_edge(out_edge, in_edge, color='black')

    for in_edge, out_edge in enumerate(network.enable_by):
        if out_edge != -1:
            G.add_edge(out_edge, in_edge, color='red')

    nodes_size = [nodes_size[n] for n in G.nodes]
    nodes_color = [nodes_color[n] for n in G.nodes]
    connections_arc = {}
    for (e1, e2) in G.edges:
        key = ((-np.sign(e1 - e2) * (1 / np.arange(1, np.abs(e1 - e2) + 1)).sum()) - 1)/10
        if key not in connections_arc:
            connections_arc[key] = []
        connections_arc[key].append((e1, e2))

    nx.draw_networkx_nodes(G,
                           pos=pos,
                           node_color=nodes_color,
                           node_size=nodes_size,
                           )
    for arc, edges in connections_arc.items():
        nx.draw_networkx_edges(G, pos, connectionstyle=f'arc3, rad = {arc}', edgelist=edges, width=2, alpha=0.5)

    def nudge(pos, x_shift, y_shift):
        return {n: (x + x_shift, y + y_shift) for n, (x, y) in pos.items()}

    pos_nodes = nudge(pos, 0, 0)
    nx.draw_networkx_labels(G, font_color="black", pos=pos_nodes, labels=nodes_labels)
