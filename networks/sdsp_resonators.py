from helpers import numbaList
from helpers.graphs import plot_network
from snn.resonator import Resonator
from snn.spiking_network import SpikingNetwork


def snn_based_resonator(frequencies):
    network = SpikingNetwork()
    clk_freq = 1.536 * 10**6
    resonators = numbaList([Resonator(freq0, clk_freq) for freq0 in frequencies])
    for resonator in resonators:
        network.add_network(resonator.network)
    return network

network = snn_based_resonator([100, 250, 700, 1520, 3700, 7500])
plot_network(network)
exit(0)

