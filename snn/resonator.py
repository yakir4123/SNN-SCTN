import json
import os

import numpy as np

from utils import njit
from snn.layers import SCTNLayer
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import IDENTITY, create_SCTN, BINARY


def create_base_resonator_by_parameters(
        freq0,
        clk_freq,
        lf,
        theta_input,
        theta,
        weight_input,
        weight_feedback,
        weight,
):
    LF = lf
    LP = lp_by_lf(LF, freq0, clk_freq)
    network = SpikingNetwork(clk_freq)
    network.add_amplitude(1000)

    # Encode to pdm
    neuron = create_SCTN()
    neuron.activation_function = IDENTITY
    network.add_layer(SCTNLayer([neuron]))

    # SCTN 1
    neuron = create_SCTN()
    neuron.synapses_weights = np.array([weight_input, -weight_feedback], dtype=np.float64)
    neuron.leakage_factor = LF
    neuron.leakage_period = LP
    neuron.theta = theta_input
    neuron.activation_function = IDENTITY
    neuron.membrane_should_reset = False
    network.add_layer(SCTNLayer([neuron]))

    for i in range(3):
        neuron = create_SCTN()
        neuron.synapses_weights = np.array([weight], dtype=np.float64)
        neuron.leakage_factor = LF
        neuron.leakage_period = LP
        neuron.theta = theta
        neuron.activation_function = IDENTITY
        neuron.membrane_should_reset = False
        network.add_layer(SCTNLayer([neuron]))

    # feedback
    network.connect_by_id(4, 1)
    return network

def _create_base_resonator_by_parameters(
        freq0,
        clk_freq,
        LF,
        LP,
        theta_gain,
        weight_gain,
        amplitude_gain
):
    if LF == -1 or LP == -1:
        LF, LP = suggest_lf_lp(freq0, clk_freq)

    network = SpikingNetwork(clk_freq)
    network.add_amplitude(1000 * amplitude_gain)
    neuron = create_SCTN()
    neuron.activation_function = IDENTITY
    network.add_layer(SCTNLayer([neuron]))

    # SCTN 1
    neuron = create_SCTN()
    neuron.synapses_weights = np.array([10 * weight_gain[0], -10 * weight_gain[1]], dtype=np.float64)
    neuron.leakage_factor = LF
    neuron.leakage_period = LP
    neuron.theta = -1 * theta_gain[0]
    neuron.activation_function = IDENTITY
    neuron.membrane_should_reset = False
    network.add_layer(SCTNLayer([neuron]))

    # SCTN 2 - 4
    for i in range(3):
        neuron = create_SCTN()
        neuron.synapses_weights = np.array([10 * weight_gain[2 + i]], dtype=np.float64)
        neuron.leakage_factor = LF
        neuron.leakage_period = LP
        neuron.theta = -5 * theta_gain[1 + i]
        neuron.membrane_should_reset = False
        neuron.activation_function = IDENTITY
        network.add_layer(SCTNLayer([neuron]))

    # feedback
    network.connect_by_id(4, 1)

    return network


def create_base_resonator(freq0, clk_freq):
    f_parameters_resonator = int(freq0 * 1536000 / clk_freq)
    with open(f'../filters2/clk_1536000/parameters/f_{f_parameters_resonator}.json') as f:
        parameters = json.load(f)
    return create_base_resonator_by_parameters(freq0, clk_freq,
                                               lf=parameters['lf'],
                                               theta_input=parameters['theta_input'],
                                               theta=parameters['theta'],
                                               weight_input=parameters['weight_input'],
                                               weight_feedback=parameters['weight_feedback'],
                                               weight=parameters['weight']
                                               )
def _create_base_resonator(freq0, clk_freq):
    with open(f'../filters/clk_{clk_freq}/parameters/f_{freq0}.json') as f:
        parameters = json.load(f)
    th_gains = [parameters[f'th_gain{i}'] for i in range(4)]
    weighted_gains = [parameters[f'weight_gain{i}'] for i in range(5)]
    return _create_base_resonator_by_parameters(freq0, clk_freq,
                                               parameters['LF'], parameters['LP'],
                                               th_gains, weighted_gains,
                                               parameters['amplitude_gain'])


def create_excitatory_resonator(freq0, clk_freq):
    network = create_base_resonator(freq0, clk_freq)

    neuron = create_SCTN()
    neuron.synapses_weights = np.array([10.0])
    neuron.leakage_period = np.inf
    neuron.theta = -4
    neuron.threshold_pulse = 50
    neuron.reset_to = 30
    neuron.activation_function = BINARY

    network.add_layer(SCTNLayer([neuron]))

    return network


def create_excitatory_inhibitory_resonator(freq0, clk_freq):
    network = SpikingNetwork(clk_freq)
    exc_resonator = create_excitatory_resonator(freq0, clk_freq)
    inh_resonator = create_excitatory_resonator(freq0, clk_freq)

    inh_resonator.neurons[0].use_clk_input = True

    network.add_network(exc_resonator)
    network.add_network(inh_resonator)

    neuron = create_SCTN()
    neuron.synapses_weights = np.array([1., -.8])
    neuron.leakage_period = np.inf
    neuron.threshold_pulse = 3
    neuron.activation_function = BINARY
    neuron.reset_to = 2
    neuron.min_clip = 0
    neuron.label = 'f' + str(freq0)

    network.add_layer(SCTNLayer([neuron]))

    return network


def trained_resonator(freq0, filters_folder='filters4_xi0'):
    #root_folder = 'C:\\Users\\user1\\PycharmProjects\\SNN-SCTN\\filters4_xi0\\united_filters'
    root_folder = 'C:\\Users\\pariz\\Project\\SNN-SCTN\\filters4_xi0\\united_filters'
    res_array = []
    for f in os.listdir(root_folder):
        if "." in (f[2:-5]):
            res_array.append(float(f[2:-5]))
        else:
            res_array.append(int(f[2:-5]))

    available_resonators = np.array(res_array)
    arg_chosen_resonator = np.argmin(np.abs(available_resonators - freq0))
    if isinstance(res_array[arg_chosen_resonator], int):
        chosen_resonator = int(available_resonators[arg_chosen_resonator])
    else:
        chosen_resonator = float(available_resonators[arg_chosen_resonator])
    print(chosen_resonator, 'Type: ', type(chosen_resonator))
    print(f'{root_folder}\\f_{chosen_resonator}.json')
    with open(f'{root_folder}\\f_{chosen_resonator}.json') as f:
        parameters = json.load(f)
        thetas = parameters['chosen_bias']
        weights = parameters['chosen_weights']
        lf = parameters['lf']
        clk_freq = parameters['clk_freq']
    print('Thetas: ', thetas, ', Weights:', weights, ', Leakage Factor:  ', lf, ', Clock Frequency: ', clk_freq)
    return simple_resonator(freq0, clk_freq, lf, thetas, weights)


def delta_resonator(freq0, filters_folder='filters4'):
    resonator = trained_resonator(freq0, filters_folder)
    neuron = create_SCTN()
    neuron.synapses_weights = np.array([3.0])
    neuron.leakage_period = np.inf
    neuron.theta = -1
    neuron.threshold_pulse = 3
    neuron.reset_to = 1.5
    neuron.activation_function = BINARY
    resonator.add_layer(SCTNLayer([neuron]))

    neuron = create_SCTN()
    neuron.synapses_weights = np.array([10.0])
    neuron.leakage_period = np.inf
    neuron.theta = 0
    neuron.leakage_factor = 1
    neuron.leakage_period = 1
    neuron.threshold_pulse = 30
    neuron.reset_to = 10
    neuron.activation_function = BINARY
    resonator.add_layer(SCTNLayer([neuron]))
    return resonator


def simple_resonator(
        freq0,
        clk_freq,
        lf,
        thetas,
        weights,
):
    LF = lf
    LP = lp_by_lf(LF, freq0, clk_freq)
    print('f is: ', freq0, 'f_clk =', clk_freq ,'Leakage Period is:',LP)
    network = SpikingNetwork()##(clk_freq)
    network.add_amplitude(1000)

    # Encode to pdm
    neuron = create_SCTN()
    neuron.activation_function = IDENTITY
    network.add_layer(SCTNLayer([neuron]))

    # SCTN 1
    neuron = create_SCTN()
    neuron.synapses_weights = np.array([weights[0], -weights[1]], dtype=np.float64)
    neuron.leakage_factor = LF
    neuron.leakage_period = LP
    neuron.theta = thetas[0]
    neuron.activation_function = IDENTITY
    neuron.membrane_should_reset = False
    network.add_layer(SCTNLayer([neuron]))

    for i in range(3):
        neuron = create_SCTN()
        neuron.synapses_weights = np.array([weights[2+i]], dtype=np.float64)
        neuron.leakage_factor = LF
        neuron.leakage_period = LP
        neuron.theta = thetas[1+i]
        neuron.activation_function = IDENTITY
        neuron.membrane_should_reset = False
        network.add_layer(SCTNLayer([neuron]))

    # feedback
    network.connect_by_id(4, 1)
    return network

@njit
def test_resonator_on_chirp(network, test_size=10_000_000, start_freq=0, clk_freq=1536000, step=None, amplifier=1):
    step = step or 1/clk_freq
    batch_size = 50_000
    shift = 0
    while test_size > 0:
        sine_size = min(batch_size, test_size)
        sine_wave, freqs = create_chirp_signal(sine_size, clk_freq, start_freq, step, shift)

        network.input_full_data(amplifier*sine_wave)

        shift = freqs[-1]
        start_freq += sine_size * step
        test_size -= sine_size


@njit
def create_chirp_signal(test_size, clk_freq, start_freq, step, shift):
    sine_wave = (np.arange(test_size) * step + start_freq + step)
    sine_wave = sine_wave * 2 * np.pi / clk_freq
    sine_wave = np.cumsum(sine_wave) + shift  # phase
    return np.sin(sine_wave), sine_wave



@njit
def freq_of_resonator(clk_freq, LF, LP):
    return clk_freq / ((2 ** LF) * 2 * np.pi * (1 + LP))


def all_lf_lp_options(lf_size, lp_size, clk_freq):
    x = np.arange(lf_size)
    y = np.arange(lp_size)
    freqs_options = np.zeros((len(x), len(y)))
    for i in range(3, len(x)):
        freqs_options[i, :] = freq_of_resonator(clk_freq, i, y)
    freqs_options[:, 0] = 0
    return freqs_options


def lp_by_lf(lf, freq0, clk_freq):
    return int((clk_freq / ((2 ** lf) * 2 * np.pi * freq0)) - 1)


def lf_lp_options(freq0, clk_freq):
    freqs_options = all_lf_lp_options(10, 400, clk_freq)
    # find the parameter that will give the closest frequency as the desired frequency
    indices = np.argmin(np.abs(freqs_options - freq0), axis=1)
    usable_lf_lp_options = np.array(list(zip(np.arange(10), indices)))
    best_lp_option = np.array([freqs_options[int(opt[0]), int(opt[1])] for opt in usable_lf_lp_options])
    res = np.zeros((len(best_lp_option), 3))
    res[:, :2] = usable_lf_lp_options
    res[:, 2] = best_lp_option
    return res


@njit
def suggest_lf_lp(freq0, f_pulse):
    _lf_lp_options = lf_lp_options(freq0, f_pulse)
    all_lf_lp_options = _lf_lp_options[:, :2]
    best_lp_option = _lf_lp_options[:, 2]
    serach_best_lp_option = np.abs(freq0 - best_lp_option) / freq0
    serach_best_lp_option = serach_best_lp_option < 0.05
    serach_best_lp_option = serach_best_lp_option[::-1]
    if sum(serach_best_lp_option) > 0:
        lp = len(serach_best_lp_option) - np.argmax(serach_best_lp_option) - 1
    else:
        lp = np.argmin(np.abs(freq0 - best_lp_option))
    lf = all_lf_lp_options[lp][0]
    lp = all_lf_lp_options[lp][1]
    return int(lf), int(lp)


def print_lf_lp_options(freq0, f_pulse):
    print(lf_lp_options(freq0, f_pulse))
