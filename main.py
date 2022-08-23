import numpy as np
from scipy import stats

from helpers import *
from helpers.graphs import plot_network
from snn.resonator import test_frequency, freq_of_resonator, \
    CustomResonator, OptimizationResonator, suggest_lf_lp, lf_lp_options
from snn.spiking_neuron import IDENTITY, createEmptySCTN


@njit
def identity_test():
    theta_step = 5
    thetas = range(0, 600, theta_step)
    # thetas = [0, 5]
    res_gauss = np.zeros((len(thetas), 1000))
    res_spikes = np.zeros(len(thetas))
    for theta in thetas:
        print(theta)
        neuron = createEmptySCTN()
        neuron.activation_function = IDENTITY
        neuron.theta = theta
        neuron.leakage_period = 1001
        for i in range(1000):
            neuron.ctn_cycle(np.array([0]), True)
            res_gauss[theta // theta_step, i] = neuron.rand_gauss_var
        res_spikes[theta // theta_step] = np.sum(neuron.out_spikes[:1000])
    return res_gauss, res_spikes


if __name__ == '__main__':
    freq0 = 2546
    LF = 4
    LP = 5

    start_freq = 0
    spectrum = 2 * freq0
    step = 1 / 10_000
    f_pulse = 1.536 * (10 ** 6)
    _lf_lp_options = lf_lp_options(freq0, f_pulse)
    _lf_lp_options_indices = abs(_lf_lp_options[:, 2] - freq0) / freq0 < 0.1
    _lf_lp_options = _lf_lp_options[_lf_lp_options_indices]
    test_size = int(spectrum / step)

    print(f'f: {freq0}, spectrum: {spectrum}, test_size: {test_size}, step: 1/{test_size // spectrum}')
    gain_factor = 9344 / ((2 ** (2 * LF - 3)) * (1 + LP))
    gains = {'th_gain0': gain_factor, 'th_gain1': gain_factor, 'th_gain2': gain_factor, 'th_gain3': gain_factor,
             'weight_gain0': gain_factor * 1.1, 'weight_gain1': gain_factor * 0.9,
             'weight_gain2': gain_factor, 'weight_gain3': gain_factor, 'weight_gain4': gain_factor,
             'amplitude_gain': gain_factor}
    # optimize by filter that generated from the output
    # gains = {'th_gain0': 181.87251160487887, 'th_gain1': 27.928785957829085, 'th_gain2': 92.84350454311262, 'th_gain3': 202.3092428465408, 'weight_gain0': 65.24512144109501, 'weight_gain1': 62.49482329934872, 'weight_gain2': 154.37826839653175, 'weight_gain3': 102.44466843729042, 'weight_gain4': 124.80108396000935, 'amplitude_gain': 196.65021930044932}
    th_gains = [gains[f'th_gain{i}'] for i in range(4)]
    weighted_gains = [gains[f'weight_gain{i}'] for i in range(5)]
    my_resonator = OptimizationResonator(freq0, f_pulse, LF, LP, th_gains, weighted_gains, gains['amplitude_gain'])
    # my_resonator = OptimizationResonator(freq0, f_pulse, -1, -1, th_gains, weighted_gains, gains['amplitude_gain'])
    # my_resonator = CustomResonator(freq0, f_pulse, LF, LP, theta_gain=-1, weight_gain=-1, amplitude_gain=-1)
    # my_resonator = Resonator(freq0, f_pulse)
    # plot_network(my_resonator.network)
    my_resonator.network.log_membrane_potential(-1)
    timing(test_frequency)(my_resonator, start_freq=start_freq, step=step, test_size=test_size)
    for i in [-1]:
        neuron = my_resonator.network.neurons[1]
        LF = neuron.leakage_factor
        LP = neuron.leakage_period
        neuron = my_resonator.network.neurons[i]

        y = neuron.membrane_potential_graph()
        x = np.linspace(start_freq, start_freq + spectrum, len(y))
        y -= np.min(y)
        max_y = np.max(y)
        y /= max_y
        plt.plot(x, y)
        f_filter = generate_sinc_filter(freq0,
                                        start_freq=start_freq,
                                        spectrum=spectrum,
                                        points=len(y),
                                        lobe_wide=600)
        # with open('filters/filter_2777.npy', 'wb') as filter_file:
        #     np.save(filter_file, y)

        # f_filter *= np.max(y) - np.min(y)
        # f_filter += np.min(y)
        plt.plot(x, f_filter)
        plt.axvline(x=freq0, c='red')
        f = int(freq_of_resonator(f_pulse, LF, LP))
        plt.title(f'neuron {i}, LF = {LF}, LP = {LP}, df = {freq0}, f = {f}')
        print(f'MSE: {sum((f_filter - y) ** 2)}')
        plt.show()
    print("Nice")
