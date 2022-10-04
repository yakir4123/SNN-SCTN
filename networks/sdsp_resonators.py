import json

import numpy as np
from matplotlib import pyplot as plt

from helpers import numbaList
from helpers.graphs import plot_network
from snn.layers import SCTNLayer
from snn.learning_rules.stdp import STDP
from snn.resonator import Resonator, CustomResonator
from snn.resonator import Resonator, CustomResonator, OptimizationResonator
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import BINARY, createEmptySCTN, SIGMOID


def snn_based_resonator(frequencies):
    network = SpikingNetwork()
    clk_freq = int(1.536 * 10 ** 6)

    for freq0 in frequencies:
        with open(f'../filters/clk_{clk_freq}/parameters/f_{freq0}.json') as f:
            parameters = json.load(f)
        th_gains = [parameters[f'th_gain{i}'] for i in range(4)]
        weighted_gains = [parameters[f'weight_gain{i}'] for i in range(5)]
        resonator = OptimizationResonator(freq0, clk_freq,
                                          parameters['LF'], parameters['LP'],
                                          th_gains, weighted_gains,
                                          parameters['amplitude_gain'])
        # resonators.append(resonator)
        # resonators = numbaList([CustomResonator(freq0, clk_freq, LF, LP) for (freq0, LF, LP) in frequencies])
        # for resonator in resonators:
        network.add_network(resonator.network)

    return network


def snn_based_resonator_for_learning(frequencies, clk_freq):
    network = snn_based_resonator(frequencies)
    neuron = createEmptySCTN()
    neuron.synapses_weights = np.random.random(len(frequencies)) * 50 + 50
    neuron.leakage_factor = 1
    neuron.leakage_period = 500
    neuron.theta = 1
    neuron.threshold_pulse = 25000
    neuron.activation_function = BINARY
    tau = 0.02  # 20 ms
    neuron.set_stdp(0.000001, 0.000001, tau, clk_freq, 150, 0)

    network.add_layer(SCTNLayer([neuron]), True, True)

    return network


def labeled_sctn_neuron(synapses_weights):
    neuron = createEmptySCTN()
    neuron.synapses_weights = synapses_weights
    neuron.leakage_factor = 1
    neuron.leakage_period = 1
    neuron.theta = 0
    neuron.threshold_pulse = 300
    neuron.activation_function = BINARY
    return neuron


def bottle_sctn_neuron():
    return labeled_sctn_neuron(np.array([53.89866944, 67.00697604, -73.62229865, -5.81576998, -53.15751697,
                                         -9.60833318, -38.93982853, -44.8558205, -28.30142606, -20.12474913,
                                         -21.33941269, 11.87871873, -66.62437808, -48.5158614, -12.55981448,
                                         -17.61929173, -59.88717952, -22.05992219, 5.40754008, 72.55390892,
                                         44.53077906, 73.34653236, -31.61805125]))


def buzzer_sctn_neuron():
    return labeled_sctn_neuron(np.array([-52.28171523, 54.9994945, -50.07000956, 16.27402393, 7.68876177,
                                         -35.21311903, 58.69939809, -14.40391217, 56.6755111, 28.7107778,
                                         10.8173887, -39.28789753, 48.71975315, -8.67652607, -47.05410993,
                                         60.71426781, 5.36805145, -57.55845561, -3.22554643, -5.27792,
                                         -69.33790142, 26.98454524, 52.23488278]))


def bell_sctn_neuron():
    return labeled_sctn_neuron(np.array([10.11675777, 65.55330472, 45.11890544, 37.62505345, -30.04633302,
                                         -18.83022219, -14.67669314, -26.49694207, -13.1400204, 13.20328768,
                                         -27.19277072, 50.69158506, 50.30866057, -18.74895392, 38.64405408,
                                         -32.48698995, 66.95057277, 25.6861972, -66.7330176, -17.27678777,
                                         -3.40383316, 18.40088793, 41.450459, ]))


def snn_based_resonator_for_test(frequencies):
    network = snn_based_resonator(frequencies)
    network.add_layer(SCTNLayer([bell_sctn_neuron(),
                                 bottle_sctn_neuron(),
                                 buzzer_sctn_neuron()]), True, True)
    return network
