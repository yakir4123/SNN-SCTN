import numpy as np

from helpers import numbaList
from helpers.graphs import plot_network
from snn.layers import SCTNLayer
from snn.resonator import Resonator, CustomResonator
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import BINARY, createEmptySCTN


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
    neuron.theta = -1
    neuron.threshold_pulse = 150000
    neuron.activation_function = BINARY
    network.add_layer(SCTNLayer([neuron]), True, True)

    return network


network = snn_based_resonator([100, 250, 700, 1520, 3700, 7500])
plot_network(network)
exit(0)

