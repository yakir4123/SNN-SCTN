from collections import OrderedDict

import numpy as np
from numba import int8, int32

from helpers.graphs import DirectedEdgeListGraph
from snn.layers import SCTNLayer
from snn.spiking_neuron import SCTNeuron, createEmptySCTN

from helpers import *


@jitclass(OrderedDict([
    ('enable_by', numbaListType(int32)),
    ('neurons', numbaListType(SCTNeuron.class_type.instance_type)),
    ('spikes_graph', DirectedEdgeListGraph.class_type.instance_type),
    ('layers_neurons', numbaListType(SCTNLayer.class_type.instance_type)),
]))
class SpikingNetwork:

    def __init__(self):
        """
        @enable_by: list of all neurons that map if neuron is enabled by other neuron
            id is enabled by self.enable_by[id]
        @spikes_graph: DirectedEdgeListGraph graph that map connections between neurons
            AND the spikes that enter the neurons
        @neurons: list of all the neurons inside the network
        @layers_neurons: list of all the layers
        """
        self.enable_by = numbaList([np.int32(0) for _ in range(0)])
        self.spikes_graph = DirectedEdgeListGraph()
        # numba needs to identify what the list type, so create empty list
        self.neurons = numbaList([createEmptySCTN() for _ in range(0)])
        self.layers_neurons = numbaList([SCTNLayer(None) for _ in range(0)])

    def add_layer(self, layer, add_neurons, connect_neurons):
        for new_neuron in layer.neurons:
            if add_neurons:
                self.add_neuron(new_neuron)
            if connect_neurons and len(self.layers_neurons) > 0:
                [self.spikes_graph.connect(neuron, new_neuron) for neuron in self.layers_neurons[-1].neurons]
        self.layers_neurons.append(layer)
        return self

    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        self.spikes_graph.add_node(neuron)
        self.enable_by.append(-1)

    def add_network(self, network):
        new_id_offset = self.spikes_graph.add_graph(network.spikes_graph)
        for neuron in network.neurons:
            neuron._id += new_id_offset
            self.neurons.append(neuron)

        # self.neurons += network.neurons
        for i in range(len(network.enable_by)):
            if network.enable_by[i] != -1:
                network.enable_by[i] += new_id_offset
                self.enable_by.append(network.enable_by[i])

        # self.enable_by += network.enable_by
        for i in range(len(network.layers_neurons)):
            if i == len(self.layers_neurons):
                self.layers_neurons.append(network.layers_neurons[i])
            else:
                self.layers_neurons[i].merge(network.layers_neurons[i])

    def get_layer(self, i):
        return self.layers_neurons[i]

    def connect(self, source_neuron, target_neuron):
        self.connect_by_id(source_neuron._id, target_neuron._id)

    def connect_by_id(self, source_id, target_id):
        self.spikes_graph.connect_by_id(source_id, target_id)

    def connect_enable_by_id(self, source_id, target_id):
        self.enable_by[target_id] = source_id

    def input(self, spike_train):
        # first update that input neurons send spikes
        self.spikes_graph.update_spike(self.neurons[0], spike_train[0])
        for neuron in self.neurons:
            enable = self.is_enable(neuron)
            emit_spike = neuron.ctn_cycle(self.spikes_graph.get_input_spikes_to(neuron), enable, False)
            self.spikes_graph.update_spike(neuron, emit_spike)

    def is_enable(self, neuron):
        # if nothing connected to enable gate or a there was a spike from the neuron that connected to enable gate
        return self.enable_by[neuron._id] == -1 or self.spikes_graph.spikes[self.enable_by[neuron._id]] == 1