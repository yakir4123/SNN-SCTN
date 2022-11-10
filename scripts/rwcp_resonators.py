import json

import numpy as np

from snn.layers import SCTNLayer
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import BINARY, create_SCTN
from snn.resonator import create_excitatory_inhibitory_resonator


def snn_based_resonator(frequencies, clk_freq):
    network = SpikingNetwork(clk_freq)

    for freq0 in frequencies:
        # resonator = create_excitatory_resonator(freq0=freq0, clk_freq=clk_freq)
        resonator = create_excitatory_inhibitory_resonator(freq0=freq0, clk_freq=clk_freq)
        network.add_network(resonator)

    return network


def create_neuron_for_labeling(synapses_weights):
    neuron = create_SCTN()
    neuron.synapses_weights = synapses_weights
    neuron.leakage_period = 3
    neuron.leakage_period = 10
    neuron.theta = 0
    neuron.threshold_pulse = 50
    neuron.activation_function = BINARY
    return neuron


def snn_based_resonator_for_learning(frequencies, clk_freq):
    network = snn_based_resonator(frequencies, clk_freq)
    synapses_weights = np.random.random(len(frequencies)) * 15 + 15
    neuron = create_neuron_for_labeling(synapses_weights)

    tau = 12 / clk_freq
    neuron.set_stdp(0.00005, 0.00008, tau, clk_freq, 200, 0)

    network.add_layer(SCTNLayer([neuron]), True, True)

    return network


def labeled_sctn_neuron(label: str):
    synapses_generations = np.load(f'neurons_weights/{label}_synapses_weights_generations.npz')
    synapses_weights = synapses_generations['synapses_weights']
    synapses_weights = synapses_weights[-1]

    neuron = create_neuron_for_labeling(synapses_weights)
    neuron.label = label
    return neuron


def snn_based_resonator_for_test(frequencies, clk_freq):
    network = snn_based_resonator(frequencies, clk_freq)
    coded_layer = SCTNLayer([
        labeled_sctn_neuron('bells5'),
        labeled_sctn_neuron('bottle1'),
        labeled_sctn_neuron('buzzer'),
        labeled_sctn_neuron('phone4'),
    ])
    network.add_layer(coded_layer, True, True)
    return network
