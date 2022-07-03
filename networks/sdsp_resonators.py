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
    resonators = numbaList([CustomResonator(freq0, clk_freq) for freq0 in frequencies])
    for resonator in resonators:
        network.add_network(resonator.network)

    neuron = createEmptySCTN()
    # neuron.synapses_weights = np.random.random(len(frequencies))
    neuron.synapses_weights = np.ones(len(frequencies))
    neuron.leakage_factor = 1
    neuron.leakage_period = 1
    neuron.theta = 0
    neuron.threshold_pulse = 15000
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
    # neuron.learning = True

    network.add_layer(SCTNLayer([neuron]), True, True)

    return network

freqs = [
        (100, 3, 299),
        (250, 4, 60),
        (500, 5, 14),
        (1000, 6, 3),
        (1750, 6, 1),
        (2800, 3, 10),
        (3500, 3, 8),
        (5000, 3, 4),
        (7500, 4, 1),
        (10000, 3, 2),
        (15000, 3, 1),
    ]
network = snn_based_resonator(freqs)
plot_network(network)

neuron = network.neurons[-1]
membrane = neuron.membrane_potential_graph[:neuron.index]
y = membrane
plt.plot(y)
plt.show()

exit(0)

