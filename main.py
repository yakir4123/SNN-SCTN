import numpy as np
from scipy import stats

from helpers import *
from helpers.graphs import plot_network
from snn.resonator import test_frequency, freq_of_resonator, \
    CustomResonator, OptimizationResonator
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
    # freq0 = 2777
    freq0 = 2777
    start_freq = 0
    spectrum = 2 * freq0
    step = 1 / 40_000
    # step = 1 / (20 * freq0)
    LF = 3
    LP = 10
    # LF = -1
    # LP = -1
    f_pulse = 1.536 * (10 ** 6)
    # test_size = 1_000_000_000
    test_size = int(spectrum / step)
    # step = 1 / (test_size // spectrum)
    print(f'f: {freq0}, spectrum: {spectrum}, test_size: {test_size}, step: 1/{test_size // spectrum}')
    gains = {'th_gain0': 1., 'th_gain1': 1., 'th_gain2': 1., 'th_gain3': 1., 'weight_gain0': 1.1, 'weight_gain1': 0.9,
             'weight_gain2': 1., 'weight_gain3': 1., 'weight_gain4': 1., 'amplitude_gain': 1.0}
    # optimize by filter that generated from the output
    # gains = {'amplitude_gain': 1.9191575383250754, 'th_gain0': 1.558377885188216, 'th_gain1': 0.4399473616072731, 'th_gain2': 1.942422205282769, 'th_gain3': 0.886318156054487, 'weight_gain0': 1.0369400177212895, 'weight_gain1': 0.7858253332474463, 'weight_gain2': 0.5317121663655087, 'weight_gain3': 1.5821842334622558, 'weight_gain4': 1.3666824669559008}
    # optimized by sinc that similar to output
    # gains = {'amplitude_gain': 1.7829775835112724, 'th_gain0': 1.4071121141018526, 'th_gain1': 0.8294516945006966, 'th_gain2': 1.9469354488769657, 'th_gain3': 1.5232168416428415, 'weight_gain0': 0.5303565105698702, 'weight_gain1': 0.3503937447731905, 'weight_gain2': 1.11292096340748, 'weight_gain3': 1.6787292645091545, 'weight_gain4': 1.4615273200986583}
    # optimized by sinc with x4 selectivity
    gains = {'th_gain0': 31.139853935349358, 'th_gain1': 22.285974684124554, 'th_gain2': 78.312515921475, 'th_gain3': 162.35960577625812, 'weight_gain0': 134.09831393975438, 'weight_gain1': 23.9062611406667, 'weight_gain2': 42.99309173241366, 'weight_gain3': 23.750009909345863, 'weight_gain4': 163.4021891890516, 'amplitude_gain': 61.12711210129383}
    th_gains = [gains[f'th_gain{i}'] for i in range(4)]
    weighted_gains = [gains[f'weight_gain{i}'] for i in range(5)]
    my_resonator = OptimizationResonator(freq0, f_pulse, LF, LP, th_gains, weighted_gains, gains['amplitude_gain'])
    # my_resonator = OptimizationResonator(freq0, f_pulse, -1, -1, th_gains, weighted_gains, gains['amplitude_gain'])
    # my_resonator = CustomResonator(freq0, f_pulse, LF, LP, -1, -1, -1)
    # my_resonator = CustomResonator(freq0, f_pulse, LF, LP, theta_gain=-1, weight_gain=-1, amplitude_gain=-1)
    # my_resonator = Resonator(freq0, f_pulse)
    # plot_network(my_resonator.network)
    my_resonator.network.log_membrane_potential(-1)
    # my_resonator.network.log_out_spikes(-1)
    timing(test_frequency)(my_resonator, start_freq=start_freq, step=step, test_size=test_size)
    for i in [-1]:
        neuron = my_resonator.network.neurons[1]
        LF = neuron.leakage_factor
        LP = neuron.leakage_period
        neuron = my_resonator.network.neurons[i]

        y = neuron.membrane_potential_graph()
        x = np.linspace(start_freq, start_freq + spectrum, len(y))
        plt.plot(x, y)
        f_filter = generate_sinc_filter(freq0, start_freq=start_freq, spectrum=spectrum,
                                        points=len(y), lobe_wide=375)

        f_filter *= np.max(y) - np.min(y)
        f_filter += np.min(y)
        plt.plot(x, f_filter)
        plt.axvline(x=freq0, c='red')
        f = int(freq_of_resonator(f_pulse, LF, LP))
        plt.title(f'neuron {i}, LF = {LF}, LP = {LP}, df = {freq0}, f = {f}')
        plt.show()
    print("Nice")
