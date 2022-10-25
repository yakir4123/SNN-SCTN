from collections import OrderedDict

import numpy as np
from numba.core.types import int32
from helpers import *


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


@njit
def njit_empty_list():
    return [np.array([np.int32(0)]) for _ in range(0)]


@njit
def njit_empty_array():
    return np.array([np.int32(0) for _ in range(0)])


def plot_network(network):
    G = nx.DiGraph()
    edges = []
    pos = {}

    column_length = max(len(layer.neurons) for layer in network.layers_neurons)
    for i, layer in enumerate(network.layers_neurons):
        for j, neuron in enumerate(layer.neurons):
            gap = column_length/len(layer.neurons)
            label = neuron.label or neuron._id
            pos[label] = [i, j * gap + gap/2]

    for out_edge, in_edges in enumerate(network.spikes_graph.out_edges):
        for in_edge in in_edges:
            label_source = network.neurons[out_edge].label or out_edge
            label_target = network.neurons[in_edge].label or in_edge

            G.add_edge(label_source, label_target, color='black')

    for in_edge, out_edge in enumerate(network.enable_by):
        if out_edge != -1:
            label_source = network.neurons[out_edge].label or out_edge
            label_target = network.neurons[in_edge].label or in_edge
            G.add_edge(label_source, label_target, color='red')

    colors = nx.get_edge_attributes(G, 'color').values()

    plt.subplot(111)
    nx.draw(G, with_labels=True, font_weight='bold', pos=pos, edge_color=colors)

    plt.show()
