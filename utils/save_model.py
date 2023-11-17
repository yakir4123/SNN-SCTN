import pickle

import numpy as np

from snn.graphs import DirectedEdgeListGraph
from snn.layers import SCTNLayer
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import create_SCTN
from utils import numbaList


def save_model(network, path):
    with open(path, 'wb') as handle:
        pickle.dump(network_to_dict(network), handle, protocol=pickle.HIGHEST_PROTOCOL)


def network_to_dict(network):
    return {
        'amplitude': network.amplitude,
        'enable_by': list(network.enable_by),
        'spikes_graph': graph_to_dict(network.spikes_graph),
        'layers_neurons': [
            layer_to_list(layer)
            for layer in network.layers_neurons
        ],
    }


def graph_to_dict(graph):
    return {
        'out_edges': list(graph.out_edges),
        'in_edges': list(graph.in_edges),
        'spikes': graph.spikes,
    }


def layer_to_list(layer):
    return [
        neuron_to_dict(n)
        for n in layer.neurons
    ]


def neuron_to_dict(neuron):
    return {
        '_id': neuron._id,
        'label': neuron.label,
        'theta': neuron.theta,
        'reset_to': neuron.reset_to,
        'use_clk_input': neuron.use_clk_input,
        'identity_const': neuron.identity_const,
        'leakage_factor': neuron.leakage_factor,
        'leakage_period': neuron.leakage_period,
        'threshold_pulse': neuron.threshold_pulse,
        'synapses_weights': neuron.synapses_weights,
        'membrane_potential': neuron.membrane_potential,
        'activation_function': neuron.activation_function,
        'membrane_should_reset': neuron.membrane_should_reset,
    }


def load_model(path):
    with open(path, 'rb') as handle:
        network_dictionary = pickle.load(handle)
    network = SpikingNetwork(network_dictionary['clk_freq'])
    network.amplitude = network_dictionary['amplitude']
    network.amplitude = network_dictionary['amplitude']
    for layer_list in network_dictionary['layers_neurons']:
        # first create neurons as templates, adding neurons change their id by connecting them to the network,
        neurons = [create_SCTN() for _ in layer_list]
        network.add_layer(SCTNLayer(neurons))

    network.enable_by = numbaList([np.int32(e) for e in network_dictionary['enable_by']])

    # now replace the connection graph to the saved graph.
    network.spikes_graph = DirectedEdgeListGraph()
    network.spikes_graph.spikes = network_dictionary['spikes_graph']['spikes']
    # return network_dictionary['spikes_graph']['out_edges']
    network.spikes_graph.out_edges = numbaList(network_dictionary['spikes_graph']['out_edges'])
    network.spikes_graph.in_edges = numbaList(network_dictionary['spikes_graph']['in_edges'])

    # now i can change the id to each neuron
    for layer_i, layer in enumerate(network.layers_neurons):
        layer_dict = network_dictionary['layers_neurons'][layer_i]
        for neuron_i, neuron in enumerate(layer.neurons):
            neuron_dict = layer_dict[neuron_i]

            neuron._id = neuron_dict['_id']
            neuron.label = neuron_dict['label']
            neuron.theta = neuron_dict['theta']
            neuron.reset_to = neuron_dict['reset_to']
            neuron.use_clk_input = neuron_dict['use_clk_input']
            neuron.identity_const = neuron_dict['identity_const']
            neuron.leakage_factor = neuron_dict['leakage_factor']
            neuron.leakage_period = neuron_dict['leakage_period']
            neuron.threshold_pulse = neuron_dict['threshold_pulse']
            neuron.synapses_weights = neuron_dict['synapses_weights']
            neuron.membrane_potential = neuron_dict['membrane_potential']
            neuron.activation_function = neuron_dict['activation_function']
            neuron.membrane_should_reset = neuron_dict['membrane_should_reset']
    return network
