import json

import numpy as np

from snn.layers import SCTNLayer
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import BINARY, create_SCTN
from snn.resonator import create_excitatory_inhibitory_resonator


def snn_based_resonators(frequencies, clk_freq):
    network = SpikingNetwork(clk_freq)

    for freq0 in frequencies:
        resonator = create_excitatory_inhibitory_resonator(freq0=freq0, clk_freq=clk_freq)
        network.add_network(resonator)

    return network


def create_neuron_for_labeling(synapses_weights):
    neuron = create_SCTN()
    neuron.synapses_weights = synapses_weights
    neuron.leakage_factor = 1
    neuron.leakage_period = 10 * len(synapses_weights)
    neuron.theta = 0
    neuron.threshold_pulse = 50 * len(synapses_weights)
    neuron.activation_function = BINARY
    return neuron


def learning_neuron(frequencies, clk_freq):
    synapses_weights = np.random.random(len(frequencies)) * 15 + 20
    neuron = create_neuron_for_labeling(synapses_weights)

    time_to_learn = 2.5e-3
    tau = clk_freq * time_to_learn / 2

    neuron.set_stdp(0.00008, 0.00008, tau, clk_freq, 50, -5)
    return neuron


def snn_based_resonator_for_learning(frequencies, clk_freq):
    network = snn_based_resonators(frequencies, clk_freq)
    neuron = learning_neuron(frequencies, clk_freq)
    network.add_layer(SCTNLayer([neuron]))
    return network


def labeled_sctn_neuron(label: str):
    synapses_generations = np.load(f'neurons_weights/{label}_synapses_weights_generations.npz')
    synapses_weights = synapses_generations['synapses_weights']
    synapses_weights = synapses_weights[-1]

    neuron = create_neuron_for_labeling(synapses_weights)
    neuron.label = label
    return neuron


def snn_based_resonator_for_test(frequencies, clk_freq):
    network = snn_based_resonators(frequencies, clk_freq)
    coded_layer = SCTNLayer([
        # labeled_sctn_neuron('bells5'),
        labeled_sctn_neuron('bottle1'),
        labeled_sctn_neuron('buzzer'),
        labeled_sctn_neuron('phone4'),
    ])
    network.add_layer(coded_layer)
    return network
