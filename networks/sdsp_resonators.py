import json

import numpy as np

from snn.layers import SCTNLayer
from snn.resonator import OptimizationResonator, create_custom_resonator
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import BINARY, createEmptySCTN


def snn_based_resonator(frequencies, clk_freq):
    network = SpikingNetwork()

    for freq0 in frequencies:
        with open(f'../filters/clk_{clk_freq}/parameters/f_{freq0}.json') as f:
            parameters = json.load(f)
        # th_gains = [parameters[f'th_gain{i}'] for i in range(4)]
        # weighted_gains = [parameters[f'weight_gain{i}'] for i in range(5)]
        #
        # resonator = OptimizationResonator(freq0, clk_freq,
        #                                   parameters['LF'], parameters['LP'],
        #                                   th_gains, weighted_gains,
        #                                   parameters['amplitude_gain'])
        resonator = create_custom_resonator(freq0=freq0, clk_freq=clk_freq)
        network.add_network(resonator.network)

    return network


def snn_based_resonator_for_learning(frequencies, clk_freq):
    network = snn_based_resonator(frequencies, clk_freq)
    neuron = createEmptySCTN()
    neuron.synapses_weights = np.random.random(len(frequencies)) * 50 + 50
    # neuron.synapses_weights = np.random.random(len(frequencies)) * 25 + 75
    neuron.leakage_factor = 3
    neuron.leakage_period = 10
    neuron.theta = -.8
    neuron.threshold_pulse = 4000
    neuron.activation_function = BINARY
    tau = 20 / clk_freq  # 0.02  # 20 ms
    neuron.set_stdp(0.00005, 0.00008, tau, clk_freq, 200, 0)

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
    return labeled_sctn_neuron(np.array([16.99665138612786, 10.067802972199448, 10.930770186481432, 14.418219483909008,
                                         14.600568854496965, 11.392388939586802, 8.430335845215323, 17.165601489139007,
                                         15.553616827528327, 15.280227093705635, 17.202680880768003, 9.704397937041835,
                                         14.336008919507218, 10.23243319292153, 17.064094482372525, 15.643889946098042,
                                         14.743206685326824, 14.906974974673306]))


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
