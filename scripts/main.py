import os
import json
import time

import yaml
import optuna

from pathlib import Path
from optuna.samplers import CmaEsSampler

from utils import *
from snn.spiking_neuron import IDENTITY, create_SCTN
from snn.resonator import test_resonator_on_chirp, freq_of_resonator, \
    lf_lp_options, create_excitatory_resonator, create_base_resonator_by_parameters


@njit
def identity_test():
    theta_step = 5
    thetas = range(0, 600, theta_step)
    # thetas = [0, 5]
    res_gauss = np.zeros((len(thetas), 1000))
    res_spikes = np.zeros(len(thetas))
    for theta in thetas:
        print(theta)
        neuron = create_SCTN()
        neuron.activation_function = IDENTITY
        neuron.theta = theta
        neuron.leakage_period = 1001
        for i in range(1000):
            neuron.ctn_cycle(np.array([0]), True)
            res_gauss[theta // theta_step, i] = neuron.rand_gauss_var
        res_spikes[theta // theta_step] = np.sum(neuron.out_spikes())
    return res_gauss, res_spikes


def simulate_and_plot(freq0, LF, LP, gains, spectrum,
                      start_freq=0, f_pulse=1_536_000, step=1 / 10_000,
                      f_resonator=-1, save_plt=None):
    if f_resonator == -1:
        f_resonator = freq0
    test_size = int(spectrum / step)
    th_gains = [gains[f'th_gain{i}'] for i in range(4)]
    weighted_gains = [gains[f'weight_gain{i}'] for i in range(5)]
    my_resonator = create_base_resonator_by_parameters(freq0, f_pulse, LF, LP, th_gains, weighted_gains, gains['amplitude_gain'])
    # plot_network(my_resonator.network)
    my_resonator.network.log_membrane_potential(-1)
    # my_resonator.network.log_out_spikes(-1)
    t = timing(test_resonator_on_chirp, return_res=False, return_time=True)(my_resonator, start_freq=start_freq, step=step,
                                                                            test_size=test_size, clk_freq=f_pulse)
    neuron = my_resonator.network.neurons[-1]
    LF = neuron.leakage_factor
    LP = neuron.leakage_period

    # plot membrane potential
    y = neuron.membrane_potential_graph()
    x = np.linspace(start_freq, start_freq + spectrum, len(y))
    # y -= np.min(y)
    # max_y = np.max(y)
    # y /= max_y
    plt.plot(x, y)

    # f_filter = generate_filter(f_resonator,
    #                            start_freq=start_freq,
    #                            spectrum=spectrum,
    #                            points=len(y),
    #                            lobe_wide=0.125 * f_resonator)
    # plt.plot(x, f_filter)
    plt.axvline(x=freq0, c='red')
    f = int(freq_of_resonator(f_pulse, LF, LP))
    # mse = sum((f_filter - y) ** 2)
    # plt.title(f'LF = {LF}, LP = {LP}, df = {freq0}, f = {f}, mse = {mse:.2f}, time={t:2.3f}s')
    plt.title(f'df = {freq0}, f = {f}, time={t:2.3f}s')
    if save_plt is not None:
        plt.savefig(save_plt)
    plt.show()
    return
    # plot emitted spikes
    y = neuron.out_spikes()
    y = np.convolve(y, np.ones(5000, dtype=int), 'valid')
    x = np.linspace(start_freq, start_freq + spectrum, len(y))
    plt.plot(x, y)
    max_y = np.max(y)

    f_filter = generate_filter(f_resonator,
                               start_freq=start_freq,
                               spectrum=spectrum,
                               points=len(y),
                               lobe_wide=0.125 * f_resonator) * max_y
    plt.plot(x, f_filter)
    plt.axvline(x=freq0, c='red')
    plt.title(f'Output spikes - df = {freq0}, f = {f}, time={t:2.3f}s')
    if save_plt is not None:
        output_file = Path(save_plt)
        output_file = output_file.parent / f'spikes_{output_file.name}'
        plt.savefig(output_file)
    plt.show()


def manual_parameters_plot():
    if LF == -1 or LP == -1:
        print(lf_lp_options(freq0=freq0, clk_freq=clk_pulse))
    gain_factor = 9344 / ((2 ** (2 * LF - 3)) * (1 + LP))
    gains = {'th_gain0': gain_factor, 'th_gain1': gain_factor, 'th_gain2': gain_factor, 'th_gain3': gain_factor,
             'weight_gain0': gain_factor * 1.1, 'weight_gain1': gain_factor * 0.9,
             'weight_gain2': gain_factor, 'weight_gain3': gain_factor, 'weight_gain4': gain_factor,
             'amplitude_gain': gain_factor}
    simulate_and_plot(freq0, LF, LP, gains, spectrum, start_freq, step=step, f_pulse=clk_pulse)


def optuna_study_plot(study_name, freq0, f_pulse=1_536_000):
    with open("../secret.yaml", 'r') as stream:
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


def suggest_lf_lp():
    _lf_lp_options = lf_lp_options(freq0, clk_pulse)
    _lf_lp_options_indices = np.argmin(abs(_lf_lp_options[:, 2] - freq0))
    LF, LP, f_resonator = _lf_lp_options[_lf_lp_options_indices]
    return LF, LP, f_resonator


def from_filter_json_plot(freq0):
    with open(f'filters/clk_{clk_pulse}/parameters/f_{freq0}.json') as f:
        parameters = json.load(f)
    simulate_and_plot(freq0, parameters['LF'], parameters['LP'], parameters,
                      2 * freq0, start_freq, clk_pulse, step, parameters['f_resonator'],
                      save_plt=f'filters/clk_{clk_pulse}/figures/f_{freq0}.png')


def plot_all_filters_json():
    filters_files = Path(f"filters/clk_{clk_pulse}/parameters").iterdir()
    for filter_file in filters_files:
        with open(filter_file) as f:
            parameters = json.load(f)

        simulate_and_plot(parameters['f0'], parameters['LF'], parameters['LP'], parameters,
                          2 * parameters['f0'], start_freq, clk_pulse, step, parameters['f_resonator'],
                          save_plt=f'filters/clk_{clk_pulse}/figures/f_{parameters["f0"]}.png')


def custom_resonator_output_spikes(
        freq0,
        clk_freq=int(1.536 * (10 ** 6)) * 2,
        step=1/12_000,
        save_figure=False):
    my_resonator = create_excitatory_resonator(freq0=freq0, clk_freq=clk_freq)
    # plot_network(my_resonator)
    # my_resonator = create_excitatory_inhibitory_resonator(freq0=freq0, clk_freq=clk_pulse)
    log_neuron_potentials = []
    for i in log_neuron_potentials:
        my_resonator.log_membrane_potential(i)
    my_resonator.log_out_spikes(-1)
    # plot_network(my_resonator.network)
    start_freq = 0
    spectrum = 2 * freq0
    test_size = int(spectrum / step)
    spikes_neuron = my_resonator.neurons[-1]

    spikes_neuron.membrane_sample_max_window = np.zeros(1).astype('float32')
    t = timing(test_resonator_on_chirp, return_res=False, return_time=True)(my_resonator,
                                                                            start_freq=start_freq, step=step,
                                                                            test_size=test_size, clk_freq=clk_freq)

    for i in log_neuron_potentials:
        membrane_neuron = my_resonator.neurons[i]
        y_membrane = membrane_neuron.membrane_potential_graph()
        x = np.linspace(start_freq, start_freq + spectrum, len(y_membrane))
        plt.title(f'membrane potential f={freq0}, neuron={i}')
        plt.plot(x, y_membrane)
        plt.show()

    y_spikes = spikes_neuron.out_spikes()

    # np.savez_compressed(f'output_{freq0}.npz',
    #                     membrane=y_membrane,
    #                     spikes=y_spikes)

    spikes_window_size = 5000
    y_spikes = np.convolve(y_spikes, np.ones(spikes_window_size, dtype=int), 'valid')
    x = np.linspace(start_freq, start_freq + spectrum, len(y_spikes))
    plt.title(f'spikes in window of {spikes_window_size} freq: {freq0}')
    if save_figure:
        # plt.savefig(f'../filters/clk_{clk_freq}/figures/f_{100}.PNG', bbox_inches='tight')
        plt.savefig('plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.plot(x, y_spikes)
        plt.show()


if __name__ == '__main__':
    for fname in os.listdir('../filters/clk_50000/parameters'):
        f0 = float(fname[2:-5])
        custom_resonator_output_spikes(f0, clk_freq=50000, step=1 / 500_000, save_figure=True)
        break
    time.sleep(1)
    freq0 = 3334
    LF = 2
    LP = 34

    start_freq = 0
    spectrum = 400
    step = 1 / 12_000
    clk_pulse = int(1.536 * (10 ** 6)) * 2
    test_size = int(spectrum / step)

    Path(f"filters/clk_{clk_pulse}/figures").mkdir(parents=True, exist_ok=True)
    Path(f"filters/clk_{clk_pulse}/parameters").mkdir(parents=True, exist_ok=True)
    # for f in range(10, stop=101, step=10):
    # suggest_lf_lp()
    # manual_parameters_plot()
    # optuna_study_plot(f'Study-{100 * (1.18 ** 0)}', freq0)
    # for f in [3934, 5478]:
    #     from_filter_json_plot(freq0=f)
    # from_filter_json_plot(freq0=3232)
    # plot_all_filters_json()
    # custom_resonator_output_spikes(freq0=236)
    # for i in range(19):
    #     f = int(200 * (1.18 ** i))
    #     custom_resonator_output_spikes(freq0=f)
    custom_resonator_output_spikes(freq0=200)
    print("Nice")
