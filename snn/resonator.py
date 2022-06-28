import json
from collections import OrderedDict

import numpy as np
from numba import float32, int32
from numpy import int8

from helpers import jitclass, njit
from snn.spiking_network import SpikingNetwork
from snn.layers import SCTNLayer
from snn.spiking_neuron import SCTNeuron, IDENTITY, createEmptySCTN, BINARY, SIGMOID


@jitclass(OrderedDict([
    ('network', SpikingNetwork.class_type.instance_type),
    ('gain_factor', float32),
    ('amplitude', float32),
    ('freq0', int32),
]))
class Resonator:

    def __init__(self, freq0, f_pulse):
        LF, LP = _desired_freq0_parameter(freq0, f_pulse)
        # LF, LP = 3, 3
        self.freq0 = freq0
        self.gain_factor = np.double(9344 / ((2**(2*LF-3))*(1+LP)))
        print(f'freq = {int(_freq_of_resonator(f_pulse, LF, LP))} with LF={LF}, LP={LP}, gain_factor={int(self.gain_factor*100000000)}/100000000')
        self.gain_factor = int(self.gain_factor*100000000)/100000000
        self.amplitude = 1000 * self.gain_factor
        self.network = SpikingNetwork()
        neuron = createEmptySCTN()
        neuron.activation_function = IDENTITY
        self.network.add_layer(SCTNLayer([neuron]))

        # SCTN 1
        neuron = createEmptySCTN()
        neuron.synapses_weights = np.array([11 * self.gain_factor, -9 * self.gain_factor], dtype=np.float64)
        neuron.leakage_factor = LF
        neuron.leakage_period = LP
        # neuron.theta = int(-1 * self.gain_factor)
        neuron.theta = -1 * self.gain_factor
        neuron.activation_function = IDENTITY
        self.network.add_layer(SCTNLayer([neuron]))

        # SCTN 2 - 4
        for i in range(3):
            neuron = createEmptySCTN()
            neuron.synapses_weights = np.array([10 * self.gain_factor], dtype=np.float64)
            neuron.leakage_factor = LF
            neuron.leakage_period = LP
            # neuron.theta = int(-5 * self.gain_factor)
            neuron.theta = -5 * self.gain_factor
            neuron.activation_function = IDENTITY
            self.network.add_layer(SCTNLayer([neuron]))

        # SCTN 5 - 8
        for i in range(4):
            neuron = createEmptySCTN()
            neuron.synapses_weights = np.array([10 * self.gain_factor], dtype=np.float64)
            neuron.leakage_factor = LF
            neuron.leakage_period = LP
            # neuron.theta = int(-5 * self.gain_factor)
            neuron.theta = -5 * self.gain_factor
            neuron.activation_function = IDENTITY
            self.network.add_layer(SCTNLayer([neuron]))
            # self.network.add_neuron(neuron)

        # SCTN 9 - 12
        for i in range(4):
            neuron = createEmptySCTN()
            neuron.synapses_weights = np.array([10 * self.gain_factor], dtype=np.float64)
            neuron.leakage_factor = LF
            neuron.leakage_period = LP
            neuron.theta = -5 * self.gain_factor
            neuron.activation_function = BINARY
            self.network.add_neuron(neuron)

        # SCTN 13 - 16
        for i in range(4):
            neuron = createEmptySCTN()
            neuron.synapses_weights = np.array([10 * self.gain_factor], dtype=np.float64)
            neuron.leakage_factor = LF
            neuron.leakage_period = LP
            neuron.theta = -5 * self.gain_factor
            neuron.activation_function = IDENTITY
            self.network.add_neuron(neuron)

        # SCTN 17
        neuron = createEmptySCTN()
        neuron.synapses_weights = np.array([6] * 4, dtype=np.float64)
        neuron.leakage_factor = 5
        neuron.leakage_period = 500
        neuron.theta = -12
        neuron.threshold_pulse = 150000
        neuron.activation_function = BINARY
        self.network.add_neuron(neuron)

        # feedbacks
        self.network.connect_by_id(4, 1)
        output_neurons = [8, 2, 4, 6]
        for i, nid in enumerate(output_neurons):
            self.network.connect_by_id(nid, 9 + i)
            self.network.connect_by_id(nid, 13 + i)
            self.network.connect_by_id(13 + i, 17)  # connect to the last neuron
            self.network.connect_enable_by_id(9 + i, 13 + i)


@jitclass(OrderedDict([
    ('network', SpikingNetwork.class_type.instance_type),
    ('gain_factor', float32),
    ('amplitude', float32),
    ('freq0', int32),
]))
class SemiResonator:

    def __init__(self, freq0, f_pulse):
        LF, LP = _desired_freq0_parameter(freq0, f_pulse)
        self.freq0 = freq0
        if LF > 2:
            self.gain_factor = 9344 / ((2**(2*LF-3))*(1+LP))
        else:
            self.gain_factor = 9344 / ((2**LF)*(1+LP))
        print(f'freq = {int(_freq_of_resonator(f_pulse, LF, LP))} with LF={LF}, LP={LP}')
        LP -= 2
        self.amplitude = 1000 * self.gain_factor
        self.network = SpikingNetwork()
        neuron = createEmptySCTN()
        neuron.activation_function = IDENTITY
        self.network.add_layer(SCTNLayer([neuron]))

        neuron = createEmptySCTN()
        neuron.synapses_weights = np.array([11 * self.gain_factor, -9 * self.gain_factor], dtype=np.float64)
        neuron.leakage_factor = LF
        neuron.leakage_period = LP
        neuron.theta = -1 * self.gain_factor
        neuron.activation_function = IDENTITY
        self.network.add_layer(SCTNLayer([neuron]))

        for i in range(3):
            neuron = createEmptySCTN()
            neuron.synapses_weights = np.array([10 * self.gain_factor], dtype=np.float64)
            neuron.leakage_factor = LF
            neuron.leakage_period = LP
            neuron.theta = -5 * self.gain_factor
            neuron.activation_function = IDENTITY
            self.network.add_layer(SCTNLayer([neuron]))
        # feedback
        self.network.connect(self.network.layers_neurons[-1].neurons[0],
                             self.network.layers_neurons[1].neurons[0])


@jitclass(OrderedDict([
    ('network', SpikingNetwork.class_type.instance_type),
    ('gain_factor', float32),
    ('amplitude', float32),
    ('freq0', int32),
]))
class CustomResonator:

    def __init__(self, freq0, f_pulse):
        LF, LP = _desired_freq0_parameter(freq0, f_pulse)
        self.freq0 = freq0
        if LF > 2:
            self.gain_factor = 9344 / ((2**(2*LF-3))*(1+LP))
        else:
            self.gain_factor = 9344 / ((2**LF)*(1+LP))

        self.amplitude = 1000 * self.gain_factor
        print(f'freq = {int(_freq_of_resonator(f_pulse, LF, LP))}, with LF={LF}, LP={LP}')
        self.network = SpikingNetwork()
        neuron = createEmptySCTN()
        neuron.activation_function = IDENTITY
        self.network.add_layer(SCTNLayer([neuron]))

        neuron = createEmptySCTN()
        neuron.synapses_weights = np.array([11 * self.gain_factor, -9 * self.gain_factor], dtype=np.float64)
        neuron.leakage_factor = LF
        neuron.leakage_period = LP
        neuron.theta = -1 * self.gain_factor
        neuron.activation_function = IDENTITY
        self.network.add_layer(SCTNLayer([neuron]))

        for i in range(3):
            neuron = createEmptySCTN()
            neuron.synapses_weights = np.array([10 * self.gain_factor], dtype=np.float64)
            neuron.leakage_factor = LF
            neuron.leakage_period = LP
            neuron.theta = -5 * self.gain_factor
            neuron.activation_function = IDENTITY
            self.network.add_layer(SCTNLayer([neuron]))

        neuron = createEmptySCTN()
        neuron.synapses_weights = np.array([10 * self.gain_factor], dtype=np.float64)
        neuron.leakage_factor = LF
        neuron.leakage_period = LP
        neuron.theta = -5 * self.gain_factor
        neuron.threshold_pulse = 10000
        neuron.activation_function = BINARY
        self.network.add_layer(SCTNLayer([neuron]))
        # feedback
        self.network.connect(self.network.layers_neurons[4].neurons[0],
                             self.network.layers_neurons[1].neurons[0])


@njit
def input_by_spike(resonator, spike):
    resonator.network.input(np.array([spike]))


@njit
def input_by_potential(resonator, potential):
    resonator.network.layers_neurons[0].neurons[0].membrane_potential = np.int16(potential * resonator.amplitude)
    resonator.network.input(np.array([0]))


@njit
def test_frequency(resonator, test_size=10_000_000, start_freq=0, step=1/200000, clk_freq=1536000):
    sine_wave = create_sine_wave(test_size, clk_freq, start_freq, step)
    for i, sample in enumerate(sine_wave):
        input_by_potential(resonator, sample)
        # if 140 <= i < 150:
            # print(f'{int(resonator.network.neurons[0].membrane_potential)}')
            # print(f'{int(resonator.network.neurons[0].rand_gauss_var)}')


@njit
def create_sine_wave(test_size, clk_freq, start_freq, step):
    sine_wave = (np.arange(test_size) * step + start_freq + step)
    sine_wave = sine_wave * 2 * np.pi / clk_freq
    sine_wave = np.cumsum(sine_wave)  # phase
    return np.sin(sine_wave)


@njit
def _freq_of_resonator(f_pulse, LF, LP):
    return f_pulse / ((2 ** LF) * 2 * np.pi * (1 + LP))


@njit
def _desired_freq0_parameter(freq0, f_pulse):
    x = np.arange(0, 8)
    y = np.arange(300)
    freqs_options = np.zeros((len(x), len(y)))
    for i in range(3, len(x)):
        freqs_options[i, :] = _freq_of_resonator(f_pulse, i, y)
    freqs_options[:, 0] = 0#np.zeros(len(x))
    # find the parameter that will give the closest frequency as the desired frequency
    indices = np.argmin(np.abs(freqs_options - freq0), axis=1)
    lf_lp_options = np.array(list(zip(x, indices)))
    best_lp_option = np.array([freqs_options[int(opt[0]), int(opt[1])] for opt in lf_lp_options])
    print_options = np.zeros((8, 3))
    print_options[:, :2] = lf_lp_options
    print_options[:, 2] = best_lp_option
    best_lp_option = np.abs(freq0 - best_lp_option) / freq0
    print(print_options)
    best_lp_option = best_lp_option < 0.05
    best_lp_option = best_lp_option[::-1]
    lp = len(best_lp_option) - np.argmax(best_lp_option) - 1
    lf = lf_lp_options[lp][0]
    lp = lf_lp_options[lp][1]
    return int(lf), int(lp)
    # indices = np.argmin(np.abs(freqs_options - freq0))
    # return indices // len(y), indices % len(y)


def log_membrane_potential(resonator, neurons_id=None):
    if neurons_id is None:
        neurons = range(len(resonator.network.neurons))
    elif type(neurons_id) == int:
        neurons = [neurons_id]
    else:
        neurons = neurons_id
    for i in neurons:
        resonator.network.neurons[i].log_membrane_potential = True


def log_rand_gauss_var(resonator, neurons_id=None):
    if neurons_id is None:
        neurons = range(len(resonator.network.neurons))
    elif type(neurons_id) == int:
        neurons = [neurons_id]
    else:
        neurons = neurons_id
    for i in neurons:
        resonator.network.neurons[i].log_rand_gauss_var = True


def log_ca(resonator, neurons_id=None):
    if neurons_id is None:
        neurons = range(len(resonator.network.neurons))
    elif type(neurons_id) == int:
        neurons = [neurons_id]
    else:
        neurons = neurons_id
    for i in neurons:
        resonator.network.neurons[i].log_ca = True


def log_out_spikes(resonator, neurons_id=None):
    if neurons_id is None:
        neurons = range(len(resonator.network.neurons))
    elif type(neurons_id) == int:
        neurons = [neurons_id]
    else:
        neurons = neurons_id
    for i in neurons:
        resonator.network.neurons[i].log_out_spikes = True

