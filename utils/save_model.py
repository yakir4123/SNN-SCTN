from snn.graphs import DirectedEdgeListGraph
from snn.layers import SCTNLayer
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import SCTNeuron


def network_to_dict(network: SpikingNetwork):
    return {
        'clk_freq': network.clk_freq,
        'amplitude': network.amplitude,
        'enable_by': network.enable_by,
        'neurons': neurons_that_not_in_layers,
        'spikes_graph': graph_to_dict(network.spikes_graph),
        'layers_neurons': [
            layer_to_list(layer)
            for layer in network.layers_neurons
        ],
    }


def graph_to_dict(graph: DirectedEdgeListGraph):
    return {
        'out_edges': graph.out_edges,
        'in_edges': graph.in_edges,
        'spikes': graph.spikes,
    }


def layer_to_list(layer: SCTNLayer):
    return [
        neuron_to_dict(n)
        for n in layer.neurons
    ]


def neuron_to_dict(neuron: SCTNeuron):
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
