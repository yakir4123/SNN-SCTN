import json

import numpy as np
import optuna
import yaml
from optuna.samplers import CmaEsSampler
from scipy import stats
from os import listdir
from os.path import isfile, join

from helpers import *
from helpers.graphs import plot_network
from snn.resonator import test_frequency, freq_of_resonator, \
    CustomResonator, OptimizationResonator, Resonator, lf_lp_options
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


def simulate_and_plot(freq0, LF, LP, gains, spectrum,
                      start_freq=0, f_pulse=1_536_000, step=1/10_000,
                      f_resonator=-1, save_plt=None):
    if f_resonator == -1:
        f_resonator = freq0
    test_size = int(spectrum / step)
    th_gains = [gains[f'th_gain{i}'] for i in range(4)]
    weighted_gains = [gains[f'weight_gain{i}'] for i in range(5)]
    my_resonator = OptimizationResonator(freq0, f_pulse, LF, LP, th_gains, weighted_gains, gains['amplitude_gain'])
    # plot_network(my_resonator.network)
    my_resonator.network.log_membrane_potential(-1)
    t = timing(test_frequency, return_res=False, return_time=True)(my_resonator, start_freq=start_freq, step=step, test_size=test_size)
    neuron = my_resonator.network.neurons[-1]
    LF = neuron.leakage_factor
    LP = neuron.leakage_period

    y = neuron.membrane_potential_graph()
    x = np.linspace(start_freq, start_freq + spectrum, len(y))
    y -= np.min(y)
    max_y = np.max(y)
    y /= max_y
    plt.plot(x, y)
    f_filter = generate_filter(f_resonator,
                               start_freq=start_freq,
                               spectrum=spectrum,
                               points=len(y),
                               lobe_wide=0.125 * f_resonator)
    plt.plot(x, f_filter)
    plt.axvline(x=freq0, c='red')
    f = int(freq_of_resonator(f_pulse, LF, LP))
    mse = sum((f_filter - y) ** 2)
    plt.title(f'LF = {LF}, LP = {LP}, df = {freq0}, f = {f}, mse = {mse:.2f}, time={t:2.3f}s')
    if save_plt is not None:
        plt.savefig(save_plt)
    plt.show()


def manual_parameters_plot():
    gain_factor = 9344 / ((2 ** (2 * LF - 3)) * (1 + LP))
    gains = {'th_gain0': gain_factor, 'th_gain1': gain_factor, 'th_gain2': gain_factor, 'th_gain3': gain_factor,
             'weight_gain0': gain_factor * 1.1, 'weight_gain1': gain_factor * 0.9,
             'weight_gain2': gain_factor, 'weight_gain3': gain_factor, 'weight_gain4': gain_factor,
             'amplitude_gain': gain_factor}
    simulate_and_plot(freq0, LF, LP, gains, spectrum, start_freq, step=step)


def optuna_study_plot(study_name, freq0, f_pulse=1_536_000):
    with open("secret.yaml", 'r') as stream:
        secrets = yaml.safe_load(stream)
    storage = f'postgresql://{secrets["USER"]}:{secrets["PASSWORD"]}@{secrets["ENDPOINT"]}:{secrets["PORT"]}/{secrets["DBNAME"]}'

    # optuna.delete_study(study_name=study_name, storage=storage)
    study = optuna.create_study(study_name=study_name,
                                storage=storage,
                                sampler=CmaEsSampler(seed=42),
                                direction='minimize',
                                load_if_exists=True)
    gains = study.best_params

    _lf_lp_options = lf_lp_options(freq0, f_pulse)
    _lf_lp_options_indices = abs(_lf_lp_options[:, 2] - freq0) / freq0 < 0.1
    _lf_lp_options = _lf_lp_options[_lf_lp_options_indices]
    LF, LP, f_resonator = _lf_lp_options[gains['lf_lp_option']]
    simulate_and_plot(freq0, LF, LP, gains, spectrum, start_freq, f_pulse, step, f_resonator)


def from_filter_json_plot(freq0):
    with open(f'filters/parameters/f_{freq0}.json') as f:
       parameters = json.load(f)
    simulate_and_plot(freq0, parameters['LF'], parameters['LP'], parameters,
                      2 * freq0, start_freq, f_pulse, step, parameters['f_resonator'],
                          save_plt=f'filters/figures/f_{freq0}.png')


def plot_all_filters_json():
    filters_files = [int(100 * (1.18 ** i)) for i in range(0, 20)]
    for freq0 in filters_files:
        with open(f'filters/parameters/f_{freq0}.json') as f:
           parameters = json.load(f)

        simulate_and_plot(freq0, parameters['LF'], parameters['LP'], parameters,
                          2 * freq0, start_freq, f_pulse, step, parameters['f_resonator'],
                          save_plt=f'filters/figures/f_{freq0}.png')


if __name__ == '__main__':
    freq0 = 104
    LF = 5
    LP = 72

    start_freq = 0
    spectrum = 2 * freq0
    step = 1 / 40_000
    f_pulse = 1.536 * (10 ** 6)
    test_size = int(spectrum / step)

    print(f'f: {freq0}, spectrum: {spectrum}, test_size: {test_size}, step: 1/{test_size // spectrum}')
    # manual_parameters_plot()
    # optuna_study_plot(f'Study0-{100 * (1.18 ** 0)}', freq0)
    from_filter_json_plot(freq0=2739)
    from_filter_json_plot(freq0=3232)
    #plot_all_filters_json()

    print("Nice")
