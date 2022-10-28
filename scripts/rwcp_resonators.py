import json

import numpy as np

from snn.layers import SCTNLayer
from snn.resonator import BaseResonator, create_excitatory_resonator, create_excitatory_inhibitory_resonator
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import BINARY, createEmptySCTN


def snn_based_resonator(frequencies, clk_freq):
    network = SpikingNetwork(clk_freq)

    for freq0 in frequencies:
        # resonator = create_excitatory_resonator(freq0=freq0, clk_freq=clk_freq)
        resonator = create_excitatory_inhibitory_resonator(freq0=freq0, clk_freq=clk_freq)
        network.add_network(resonator.network)

    return network


def snn_based_resonator_for_learning(frequencies, clk_freq):
    network = snn_based_resonator(frequencies, clk_freq)
    neuron = createEmptySCTN()
    neuron.synapses_weights = np.random.random(len(frequencies)) * 20 + 20
    # neuron.synapses_weights = np.random.random(len(frequencies)) * 25 + 75
    neuron.leakage_factor = 3
    neuron.leakage_period = np.inf
    neuron.theta = 0#-.02
    neuron.threshold_pulse = 20
    neuron.activation_function = BINARY
    tau = 12 / clk_freq  # 0.02  # 20 ms
    neuron.set_stdp(0.00005, 0.00008, tau, clk_freq, 200, 0)

    network.add_layer(SCTNLayer([neuron]), True, True)

    return network


def labeled_sctn_neuron(label):
    synapses_generations = np.load(f'neurons_weights/{label}_synapses_weights_generations.npz')
    synapses_weights = synapses_generations['synapses_weights'][-1]

    neuron = createEmptySCTN()
    neuron.synapses_weights = synapses_weights
    # neuron.synapses_weights = np.zeros(len(synapses_weights) + 1)
    # neuron.synapses_weights[:-1] = synapses_weights
    # neuron.synapses_weights[-1] = -2000
    neuron.label = label
    neuron.leakage_period = 8
    neuron.leakage_period = 8
    neuron.theta = -.8
    neuron.threshold_pulse = 3000
    neuron.activation_function = BINARY
    return neuron


def silence_sctn_neuron():
    synapses_weights = np.array(
        [11.658501737074129, 16.307419813600745, 14.18068504990376, 10.545513222263466, 12.651811684780517,
         15.217384338767662, 7.910698616708862, 14.549317168671944, 8.954442501782344, 9.542379518824264,
         15.09037610800621, 14.270712683921625, 11.524699438111547, 16.33539582527102, 10.008313749018436,
         18.453458749397296, 13.936458335154036, 9.987915000046055]
    )
    neuron = createEmptySCTN()
    neuron.synapses_weights = synapses_weights
    neuron.synapses_weights[-1] = -2000
    neuron.leakage_factor = 3
    neuron.leakage_period = 8
    neuron.theta = -.8
    neuron.threshold_pulse = 3000
    neuron.activation_function = BINARY
    return neuron


def snn_based_resonator_for_test(frequencies, clk_freq):
    network = snn_based_resonator(frequencies, clk_freq)
    coded_layer = SCTNLayer([
        labeled_sctn_neuron('bells5'),
        labeled_sctn_neuron('bottle1'),
        labeled_sctn_neuron('buzzer'),
        labeled_sctn_neuron('phone4'),
        # silence_sctn_neuron(),
    ])
    network.add_layer(coded_layer, True, True)
    # for neuron in coded_layer.neurons[:-1]:
    #     network.connect(coded_layer.neurons[-1], neuron)
    return network
