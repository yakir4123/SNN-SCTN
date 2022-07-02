import numpy as np

from helpers import numbaList
from helpers.graphs import plot_network
from snn.layers import SCTNLayer
from snn.resonator import Resonator, CustomResonator
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import BINARY, createEmptySCTN, SIGMOID


def snn_based_resonator(frequencies):
    network = SpikingNetwork()
    clk_freq = 1.536 * 10**6
    resonators = numbaList([CustomResonator(freq0, clk_freq) for freq0 in frequencies])
    for resonator in resonators:
        network.add_network(resonator.network)

    neuron = createEmptySCTN()
    neuron.synapses_weights = np.random.random(len(frequencies))
    neuron.leakage_factor = 1
    neuron.leakage_period = 1
    neuron.theta = 0
    neuron.threshold_pulse = 150000
    neuron.activation_function = BINARY

    # sdsp
    neuron.ca = 0
    neuron.ca_peak = 1
    neuron.max_weight = 0
    neuron.min_weight = 1
    neuron.threshold_potentiation_low = 10
    neuron.threshold_potentiation_high = 100
    neuron.threshold_depression_low = 10
    neuron.threshold_depression_high = 100
    neuron.threshold_potential = 3
    neuron.threshold_weight = 0.5 * (neuron.max_weight - neuron.min_weight)
    neuron.delta_x = 0.0000005 * neuron.max_weight

    network.add_layer(SCTNLayer([neuron]), True, True)

    return network


network = snn_based_resonator([100, 250, 500, 1000, 1750, 2800, 3500, 5000, 7500, 10000, 15000])
plot_network(network)
exit(0)

