import json
from collections import OrderedDict

import numpy as np
from numba import float32, int32

from helpers import jitclass, njit
from snn.spiking_network import SpikingNetwork
from snn.layers import SCTNLayer
from snn.spiking_neuron import SCTNeuron, IDENTITY, createEmptySCTN, BINARY, SIGMOID


@jitclass(OrderedDict([
    ('network', SpikingNetwork.class_type.instance_type),
    ('gain_factor', float32),
    ('freq0', int32),
]))
class Resonator:

    def __init__(self, freq0, f_pulse):
        LF, LP = _desired_freq0_parameter(freq0, f_pulse)
        # LF, LP = 6, 36
        self.freq0 = freq0
        self.gain_factor = np.double(9344 / ((2**(2*LF-3))*(1+LP)))
        print(f'freq = {int(freq_of_resonator(f_pulse, LF, LP))} with LF={LF}, LP={LP}, gain_factor={int(self.gain_factor * 100000000)}/100000000')
        self.gain_factor = int(self.gain_factor*100000000)/100000000
        self.network = SpikingNetwork()
        self.network.add_amplitude(1000 * self.gain_factor)
        neuron = createEmptySCTN()
        neuron.activation_function = IDENTITY
        self.network.add_neuron(neuron)

        # SCTN 1
        neuron = createEmptySCTN()
        neuron.synapses_weights = np.array([11 * self.gain_factor, -9 * self.gain_factor], dtype=np.float64)
        neuron.leakage_factor = LF
        neuron.leakage_period = LP
        neuron.theta = -1 * self.gain_factor
        neuron.activation_function = IDENTITY
        self.network.add_neuron(neuron)

        # SCTN 2 - 4
        for i in range(3):
            neuron = createEmptySCTN()
            neuron.synapses_weights = np.array([10 * self.gain_factor], dtype=np.float64)
            neuron.leakage_factor = LF
            neuron.leakage_period = LP
            neuron.theta = -5 * self.gain_factor
            neuron.activation_function = IDENTITY
            self.network.add_neuron(neuron)

        # SCTN 5 - 8
        for i in range(4):
            neuron = createEmptySCTN()
            neuron.synapses_weights = np.array([10 * self.gain_factor], dtype=np.float64)
            neuron.leakage_factor = LF
            neuron.leakage_period = LP
            neuron.theta = -5 * self.gain_factor
            neuron.activation_function = IDENTITY
            self.network.add_neuron(neuron)

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
        neuron.activation_function = IDENTITY
        self.network.add_neuron(neuron)

        for neuron in self.network.neurons:
            neuron.membrane_should_reset = False

        for i in range(0, 8):
            self.network.connect_by_id(i, i + 1)

        # feedbacks
        self.network.connect_by_id(4, 1)
        output_neurons = [8, 2, 4, 6]
        for i, nid in enumerate(output_neurons):
            self.network.connect_by_id(nid, 9 + i)
            self.network.connect_by_id(nid, 13 + i)
            self.network.connect_by_id(13 + i, 17)  # connect to the last neuron
            self.network.connect_enable_by_id(9 + i, 13 + i)

        layer0 = SCTNLayer([self.network.neurons[0]])
        layer1 = SCTNLayer([self.network.neurons[i] for i in range(1, 5)])
        layer2 = SCTNLayer([self.network.neurons[i] for i in range(5, 9)])
        layer3 = SCTNLayer([self.network.neurons[i] for i in range(9, 13)])
        layer4 = SCTNLayer([self.network.neurons[i] for i in range(13, 17)])
        layer5 = SCTNLayer([self.network.neurons[17]])
        self.network.add_layer(layer0, False, False)
        self.network.add_layer(layer1, False, False)
        self.network.add_layer(layer2, False, False)
        self.network.add_layer(layer3, False, False)
        self.network.add_layer(layer4, False, False)
        self.network.add_layer(layer5, False, False)


@jitclass(OrderedDict([
    ('network', SpikingNetwork.class_type.instance_type),
    ('gain_factor', float32),
    ('theta_gain', float32),
    ('weight_gain', float32),
    ('amplitude_gain', float32),
    ('freq0', int32),
]))
class CustomResonator:

    def __init__(self,
                 freq0,
                 f_pulse,
                 LF,
                 LP,
                 theta_gain,
                 weight_gain,
                 amplitude_gain):
        if LF == -1 or LP == -1:
            LF, LP = _desired_freq0_parameter(freq0, f_pulse)
        self.freq0 = freq0
        self.gain_factor = np.double(9344 / ((2**(2*LF-3))*(1+LP)))
        if theta_gain == -1:
            self.theta_gain = self.gain_factor
        else:
            self.theta_gain = theta_gain
        if weight_gain == -1:
            self.weight_gain = self.gain_factor
        else:
            self.weight_gain = weight_gain
        if amplitude_gain == -1:
            self.amplitude_gain = self.gain_factor
        else:
            self.amplitude_gain = amplitude_gain

        self.network = SpikingNetwork()
        self.network.add_amplitude(1000 * self.amplitude_gain)
        neuron = createEmptySCTN()
        neuron.activation_function = IDENTITY
        self.network.add_neuron(neuron)

        # SCTN 1
        neuron = createEmptySCTN()
        neuron.synapses_weights = np.array([11 * self.weight_gain, -9 * self.weight_gain], dtype=np.float64)
        neuron.leakage_factor = LF
        neuron.leakage_period = LP
        neuron.theta = -1 * self.theta_gain
        neuron.activation_function = IDENTITY
        self.network.add_neuron(neuron)

        # SCTN 2 - 4
        for i in range(3):
            neuron = createEmptySCTN()
            neuron.synapses_weights = np.array([10 * self.weight_gain], dtype=np.float64)
            neuron.leakage_factor = LF
            neuron.leakage_period = LP
            neuron.theta = -5 * self.theta_gain
            neuron.activation_function = IDENTITY
            self.network.add_neuron(neuron)

        for neuron in self.network.neurons:
            neuron.membrane_should_reset = False

        for i in range(0, 4):
            self.network.connect_by_id(i, i + 1)

        # feedback
        self.network.connect(self.network.neurons[4],
                             self.network.neurons[1])

        layer0 = SCTNLayer([self.network.neurons[0]])
        layer1 = SCTNLayer([self.network.neurons[i] for i in range(1, 4)])
        layer2 = SCTNLayer([self.network.neurons[4]])
        self.network.add_layer(layer0, False, False)
        self.network.add_layer(layer1, False, False)
        self.network.add_layer(layer2, False, False)


@jitclass(OrderedDict([
    ('network', SpikingNetwork.class_type.instance_type),
    ('gain_factor', float32),
    ('amplitude_gain', float32),
    ('freq0', int32),
]))
class OptimizationResonator:

    def __init__(self,
                 freq0,
                 f_pulse,
                 LF,
                 LP,
                 theta_gain,
                 weight_gain,
                 amplitude_gain):
        if LF == -1 or LP == -1:
            LF, LP = _desired_freq0_parameter(freq0, f_pulse)
        self.freq0 = freq0

        self.network = SpikingNetwork()
        self.network.add_amplitude(1000 * amplitude_gain)
        neuron = createEmptySCTN()
        neuron.activation_function = IDENTITY
        self.network.add_neuron(neuron)

        # SCTN 1
        neuron = createEmptySCTN()
        neuron.synapses_weights = np.array([10 * weight_gain[0], -10 * weight_gain[1]], dtype=np.float64)
        neuron.leakage_factor = LF
        neuron.leakage_period = LP
        neuron.theta = -1 * theta_gain[0]
        neuron.activation_function = IDENTITY
        self.network.add_neuron(neuron)

        # SCTN 2 - 4
        for i in range(3):
            neuron = createEmptySCTN()
            neuron.synapses_weights = np.array([10 * weight_gain[2 + i]], dtype=np.float64)
            neuron.leakage_factor = LF
            neuron.leakage_period = LP
            neuron.theta = -5 * theta_gain[1 + i]
            neuron.activation_function = IDENTITY
            self.network.add_neuron(neuron)

        for neuron in self.network.neurons:
            neuron.membrane_should_reset = False

        for i in range(0, 4):
            self.network.connect_by_id(i, i + 1)

        # feedback
        self.network.connect(self.network.neurons[4],
                             self.network.neurons[1])

        layer0 = SCTNLayer([self.network.neurons[0]])
        layer1 = SCTNLayer([self.network.neurons[i] for i in range(1, 4)])
        layer2 = SCTNLayer([self.network.neurons[4]])
        self.network.add_layer(layer0, False, False)
        self.network.add_layer(layer1, False, False)
        self.network.add_layer(layer2, False, False)


@njit
def input_by_spike(resonator, spike):
    resonator.network.input(np.array([spike]))


@njit
def input_by_potential(resonator, potential):
    # resonator.network.layers_neurons[0].neurons[0].membrane_potential = np.int16(potential * resonator.amplitude)
    # resonator.network.input(np.array([0]))
    resonator.network.input_potential(np.array([potential]))


@njit
def test_frequency(resonator, test_size=10_000_000, start_freq=0, step=1/200000, clk_freq=1536000):
    sine_wave = create_sine_wave(test_size, clk_freq, start_freq, step)
    for i, sample in enumerate(sine_wave):
        input_by_potential(resonator, sample)
        # if 0 <= i < 20:
        #     print(f'{int(resonator.network.neurons[1].membrane_potential)}')
        #     print(f'{int(resonator.network.neurons[0].rand_gauss_var)}')

@njit
def create_sine_wave(test_size, clk_freq, start_freq, step):
    sine_wave = (np.arange(test_size) * step + start_freq + step)
    sine_wave = sine_wave * 2 * np.pi / clk_freq
    sine_wave = np.cumsum(sine_wave)  # phase
    return np.sin(sine_wave)


@njit
def freq_of_resonator(f_pulse, LF, LP):
    return f_pulse / ((2 ** LF) * 2 * np.pi * (1 + LP))


@njit
def _desired_freq0_parameter(freq0, f_pulse):
    x = np.arange(0, 8)
    y = np.arange(300)
    freqs_options = np.zeros((len(x), len(y)))
    for i in range(3, len(x)):
        freqs_options[i, :] = freq_of_resonator(f_pulse, i, y)
    freqs_options[:, 0] = 0
    # find the parameter that will give the closest frequency as the desired frequency
    indices = np.argmin(np.abs(freqs_options - freq0), axis=1)
    lf_lp_options = np.array(list(zip(x, indices)))
    best_lp_option = np.array([freqs_options[int(opt[0]), int(opt[1])] for opt in lf_lp_options])
    print_options = np.zeros((8, 3))
    print_options[:, :2] = lf_lp_options
    print_options[:, 2] = best_lp_option
    print(print_options.astype(np.int32))
    serach_best_lp_option = np.abs(freq0 - best_lp_option) / freq0
    serach_best_lp_option = serach_best_lp_option < 0.05
    serach_best_lp_option = serach_best_lp_option[::-1]
    if sum(serach_best_lp_option) > 0:
        lp = len(serach_best_lp_option) - np.argmax(serach_best_lp_option) - 1
    else:
        lp = np.argmin(np.abs(freq0 - best_lp_option))
    lf = lf_lp_options[lp][0]
    lp = lf_lp_options[lp][1]
    return int(lf), int(lp)

