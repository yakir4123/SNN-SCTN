import numpy as np

from helpers import *
from helpers.graphs import plot_network
from snn.resonator import Resonator, test_frequency, log_membrane_potential, freq_of_resonator, log_out_spikes, \
    CustomResonator
from snn.spiking_neuron import IDENTITY, createEmptySCTN


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'func:{f.__name__} args:({args}, {kw}] took: {te - ts:2.4f} sec')
        return result

    return wrap


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
            res_gauss[theta//theta_step, i] = neuron.rand_gauss_var
        res_spikes[theta//theta_step] = np.sum(neuron.out_spikes[:1000])
    return res_gauss, res_spikes


if __name__ == '__main__':
    freq0 = 250
    start_freq = 0
    spectrum = 2*(freq0 - start_freq)

    f_pulse = 1.536 * (10 ** 6)
    test_size = 1_000_000
    step = 1 / (test_size // spectrum)
    print(f'f: {freq0}, spectrum: {spectrum}, test_size: {test_size}, step: 1/{test_size // spectrum}')
    my_resonator = CustomResonator(freq0, f_pulse, 4, 60)
    # my_resonator = Resonator(freq0, f_pulse)
    # plot_network(my_resonator.network)
    log_membrane_potential(my_resonator, -1)
    log_out_spikes(my_resonator, -1)
    timing(test_frequency)(my_resonator, start_freq=start_freq, step=step, test_size=test_size)
    for i in [-1]:
        neuron = my_resonator.network.neurons[1]
        LF = neuron.leakage_factor
        LP = neuron.leakage_period
        neuron = my_resonator.network.neurons[i]
        skip = 50
        spikes_amount = neuron.out_spikes[:neuron.index]
        spikes_amount = np.convolve(spikes_amount, np.ones(5000, dtype=int), 'valid')
        y = spikes_amount[::skip]
        x = np.arange(start_freq, start_freq + test_size*step, step*skip)[:len(y)]
        plt.plot(x, y)
        plt.axvline(x=freq0, c='red')
        plt.title(f'neuron {i} spikes')
        plt.show()
        membrane = neuron.membrane_potential_graph[:neuron.index]
        y = membrane[::skip]
        x = np.arange(start_freq, start_freq + test_size*step, step*skip)[:len(y)]
        plt.plot(x, y)
        plt.axvline(x=freq0, c='red')
        f = int(freq_of_resonator(f_pulse, LF, LP))
        plt.title(f'neuron {i}, LF = {LF}, LP = {LP}, df = {freq0}, f = {f}')
        plt.show()
    print("Nice")
