import json
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import numpy as np
from optuna.samplers import TPESampler

from snn.layers import SCTNLayer
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import create_SCTN, IDENTITY
from snn.resonator import test_resonator_on_chirp, lp_by_lf, freq_of_resonator

freq0 = 104
freq_base = 104
clk_freq = 1536000
duration = .6 * freq_base / freq0
spikes_window_size = 500
trials = 300
# lf_options = [5, 4, 6, 7, 3, 8]
lf_options = [5]


def neuron_output(neuron, rolling_window, duration, signal_freq, shift_degrees=0):
    y_events = neuron.out_spikes()
    cycles = 5
    samples_per_cycle =  clk_freq / signal_freq
    samples_per_degree = samples_per_cycle/360
    shift_samples = int(shift_degrees*samples_per_degree)
    y_events = y_events[
        (y_events > (duration - ((5+cycles)/signal_freq)) * clk_freq + shift_samples) &
        (y_events < (duration - (5/signal_freq)) * clk_freq + shift_samples)
    ]
    if len(y_events) == 0:
        raise ValueError("No spikes were emit.")
    y_events -= y_events[0]
    y_spikes = np.zeros(int(cycles/signal_freq * clk_freq) + 1)
    y_spikes[y_events] = 1
    y_spikes_rollsum = np.convolve(y_spikes, np.ones(rolling_window, dtype=int), 'valid')
    return y_spikes_rollsum

def generate_and_input_signal(resonator, duration, f):
    x = np.linspace(0, duration, int(duration * resonator.clk_freq))
    t = x * 2 * np.pi * f
    sine_wave = np.sin(t)
    resonator.input_full_data(sine_wave)

def resonator_from_optuna(freq0, lf, theta_input, theta, weight_input, weight_feedback, weight, amplitude_gain=1, clk_freq=1_536_000):
    LF = lf
    LP = lp_by_lf(lf, freq0, clk_freq)
    network = SpikingNetwork(clk_freq)
    network.add_amplitude(1000 * amplitude_gain)

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

def objective(trial):
    gain = max(1., freq0 / 104)
    theta_input = trial.suggest_int('theta_input', -int(gain*10), -int(gain*1))
    theta = trial.suggest_int('theta', -int(gain*10), -int(gain*1))
    weight_input = trial.suggest_int('weight_input', int(gain*5), int(gain*15))
    weight_feedback = trial.suggest_int('weight_feedback', int(gain*5), int(gain*15))
    weight = trial.suggest_int('weight', int(gain*5), int(gain*15))
    amplitude_gain=1

    resonator = resonator_from_optuna(freq0, lf, theta_input, theta, weight_input, weight_feedback, weight, amplitude_gain, clk_freq)
    return  score_resonator(resonator, duration=duration, freq0=freq0)

def score_resonator(resonator, duration, freq0):
    for i in range(0, 5):
        resonator.log_out_spikes(i)

    x = np.linspace(0, duration, int(duration * resonator.clk_freq))
    t = x * 2 * np.pi * freq0
    sine_wave = np.sin(t)
    resonator.input_full_data(sine_wave)

    try:
        spikes_roll = np.array([neuron_output(resonator.neurons[i], spikes_window_size, duration, signal_freq=freq0)
                                for i in range(0, 5)])
    except ValueError:
        return np.inf

    mse = 0
    for i in range(0, 4):
        mse += ((ground_truth[i] - spikes_roll[i+1]) ** 2).mean()
    return mse

base_params = {
    'theta_input': -1,
    'theta': -5,
    'weight_input': 11,
    'weight_feedback': 9,
    'weight': 10,
}


Path(f"../filters2/clk_{clk_freq}/figures").mkdir(parents=True, exist_ok=True)
Path(f"../filters2/clk_{clk_freq}/parameters").mkdir(parents=True, exist_ok=True)

best_params = {}
best_lf = 0
best_lf_score = np.inf
for lf in lf_options:
# for lf in range(5, 6):
    resonator = resonator_from_optuna(freq_base, **base_params, lf=5, clk_freq=clk_freq)
    desired_resonator = resonator_from_optuna(freq0, **base_params, lf=lf, clk_freq=clk_freq)
    desired_resonator.log_out_spikes(0)
    for i in range(0, 5):
        resonator.log_out_spikes(i)
    generate_and_input_signal(resonator, duration, freq_base)
    generate_and_input_signal(desired_resonator, duration, freq0)
    ground_truth_104 = np.array([neuron_output(resonator.neurons[i], spikes_window_size, duration, freq_base) for i in range(0, 5)])

    phase_shift = 0
    ground_truth = []
    for i in range(1, 5):
        j = i - 1
        norm_x0 = (ground_truth_104[j] - np.min(ground_truth_104[j])) / (np.max(ground_truth_104[j]) - np.min(ground_truth_104[j])) * 2 - 1
        max_xi = ground_truth_104[i].max()
        min_xi = ground_truth_104[i].min()
        norm_x1 = (ground_truth_104[i] - min_xi) / (max_xi - min_xi) * 2 - 1
        align_norms = np.array([norm_x1[::20], norm_x0[::20]])
        cov = np.cov(np.transpose(align_norms))
        phi = np.arccos(cov[1, 0]) * 180 / np.pi / 2
        phase_shift -= phi
        gt_i = neuron_output(desired_resonator.neurons[0], spikes_window_size, duration, freq0, phase_shift)
        gt_i = (gt_i - gt_i.min())/(gt_i.max() - gt_i.min())
        gt_i = gt_i * (max_xi - min_xi) + min_xi
        ground_truth.append(gt_i)

    ground_truth = np.array(ground_truth)
    if freq0 == freq_base:
        # for testing the mse of the ground truth vs what we are searching
        # mse = score_resonator(resonator, duration=duration, freq0=freq0)
        mse = score_resonator(resonator, duration=duration, freq0=freq0)
        print(f'mse {mse}')


    study_name = f'Study{clk_freq}-{freq0}'
    study = optuna.create_study(study_name=study_name,
                                pruner=optuna.pruners.HyperbandPruner(),
                                sampler=TPESampler(seed=43),
                                direction='minimize',
                                load_if_exists=True)

    study.optimize(objective, n_trials=trials)
    best_params[lf] = study.best_params.copy()
    if study.best_trial.value < best_lf_score:
        best_lf_score = study.best_trial.value
        best_lf = lf


start_freq = 0
spectrum = 2 * freq0
resonator = resonator_from_optuna(freq0, **best_params[best_lf], lf=best_lf, clk_freq=clk_freq)
resonator.log_out_spikes(-1)

step = 1 / 20000
test_size = int(spectrum / step)
test_resonator_on_chirp(
    resonator,
    start_freq=start_freq,
    step=step,
    test_size=test_size,
    clk_freq=clk_freq
)

spikes_neuron = resonator.neurons[-1]
y_events = spikes_neuron.out_spikes()
y_spikes = np.zeros(test_size)
y_spikes[y_events] = 1
spikes_window_size = 500
y_spikes = np.convolve(y_spikes, np.ones(spikes_window_size, dtype=int), 'valid')
x = np.linspace(start_freq, start_freq+spectrum, len(y_spikes))

best_lp = lp_by_lf(best_lf, freq0, clk_freq)
f_resonator = freq_of_resonator(clk_freq, best_lf, best_lp)

plt.title(f'spikes in window of {spikes_window_size} freq: {f_resonator:.2f} LF: {best_lf}')
plt.plot(x, y_spikes)
plt.savefig(f'../filters2/clk_{clk_freq}/figures/f_{int(freq0)}.png', bbox_inches='tight')
plt.close()

with open(f"../filters2/clk_{clk_freq}/parameters/f_{int(freq0)}.json", 'w') as best_params_f:
    best_parameters = best_params[best_lf]
    best_parameters['lf'] = best_lf
    best_parameters['lp'] = best_lp
    best_parameters['f_resonator'] = f_resonator
    best_parameters['desired_freq'] = freq0
    json.dump(best_parameters, best_params_f, indent=4)
