import numpy as np
from matplotlib import pyplot as plt

from helpers import numbaList
from helpers.graphs import plot_network
from snn.layers import SCTNLayer
from snn.resonator import Resonator, CustomResonator
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import BINARY, createEmptySCTN, SIGMOID


def snn_based_resonator(frequencies):
    network = SpikingNetwork()
    clk_freq = 1.536 * 10**6
    resonators = numbaList([CustomResonator(freq0, clk_freq, LF, LP) for (freq0, LF, LP) in frequencies])
    for resonator in resonators:
        network.add_network(resonator.network)

    return network


def snn_based_resonator_for_learning(frequencies):
    network = snn_based_resonator(frequencies)
    neuron = createEmptySCTN()
    neuron.synapses_weights = np.random.random(len(frequencies))
    neuron.leakage_factor = 1
    neuron.leakage_period = 1
    neuron.theta = 0
    neuron.threshold_pulse = 5
    neuron.activation_function = BINARY

    # sdsp
    neuron.ca = 0
    neuron.ca_peak = 1
    neuron.max_weight = 1
    neuron.min_weight = 0
    neuron.threshold_potentiation_low = 10
    neuron.threshold_potentiation_high = 100
    neuron.threshold_depression_low = 10
    neuron.threshold_depression_high = 100
    neuron.threshold_potential = 5
    neuron.threshold_weight = 0.5 * (neuron.max_weight - neuron.min_weight)
    neuron.delta_x = 0.0000005 * neuron.max_weight
    neuron.shifting_const = 0.00000008 * neuron.max_weight
    neuron.learning = True

    network.add_layer(SCTNLayer([neuron]), True, True)

    return network


def snn_based_resonator_for_test(frequencies):
    network = snn_based_resonator(frequencies)
    bells_neuron = createEmptySCTN()
    bells_neuron.synapses_weights = np.array([1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0.])
    bells_neuron.leakage_factor = 1
    bells_neuron.leakage_period = 1
    bells_neuron.theta = 0
    bells_neuron.threshold_pulse = 5
    bells_neuron.activation_function = BINARY

    bottle_neuron = createEmptySCTN()
    bottle_neuron.synapses_weights = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1.])
    bottle_neuron.leakage_factor = 1
    bottle_neuron.leakage_period = 1
    bottle_neuron.theta = 0
    bottle_neuron.threshold_pulse = 5
    bottle_neuron.activation_function = BINARY

    buzzer_neuron = createEmptySCTN()
    buzzer_neuron.synapses_weights = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1.])
    buzzer_neuron.leakage_factor = 1
    buzzer_neuron.leakage_period = 1
    buzzer_neuron.theta = 0
    buzzer_neuron.threshold_pulse = 5
    buzzer_neuron.activation_function = BINARY

    network.add_layer(SCTNLayer([bells_neuron, bottle_neuron, buzzer_neuron]), True, True)
    return network

# #1 - fail, bells domintate (4n resonator)
# bells5 [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1.]
# bottle1 [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0.]
# buzzer [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0.]

# #2 - fail, buzzer domintate (5n resonator)
# bells5 [1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0.]
# bottle1 [1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1.]
# buzzer [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1.]

