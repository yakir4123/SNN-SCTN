import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from snn.layers import SCTNLayer
from snn.resonator import lp_by_lf
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import create_SCTN, IDENTITY
from snn.resonator import test_resonator_on_chirp, freq_of_resonator


def neuron_output(neuron, signal_freq, shift_degrees=0, phase_number=20):
    y_events = neuron.out_spikes()
    samples_per_cycle =  clk_freq / signal_freq
    samples_per_degree = samples_per_cycle/360
    shift_samples = int(shift_degrees*samples_per_degree)
    y_events = y_events[
        (y_events > ((phase_number-1)/signal_freq) * clk_freq + shift_samples) &
        (y_events < ((phase_number/signal_freq) * clk_freq) + shift_samples)
    ]
    if len(y_events) == 0:
        return y_events
    return y_events


def events_to_spikes(events, run_window=0, spikes_arr_size=-1):
    if spikes_arr_size == -1:
        y_spikes = np.zeros(events[-1] + 1)
    else:
        y_spikes = np.zeros(spikes_arr_size)

    y_spikes[events] = 1
    if run_window == 0:
        return y_spikes

    y_spikes_rollsum = np.convolve(y_spikes, np.ones(run_window, dtype=int), 'valid')
    return y_spikes_rollsum

def learning_resonator(
        lf,
        freq0,
        thetas,
        weights,
        ground_truths,
        A,
        time_to_learn,
        max_weight,
        min_weight,
):
    LF = lf
    LP = lp_by_lf(LF, freq0, clk_freq)
    network = SpikingNetwork(clk_freq)
    tau = network.clk_freq * time_to_learn / 2
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
    neuron.set_supervised_stdp(A, tau, clk_freq, max_weight, min_weight, ground_truths[0])
    network.add_layer(SCTNLayer([neuron]))

    for i in range(3):
        neuron = create_SCTN()
        neuron.synapses_weights = np.array([weights[2+i]], dtype=np.float64)
        neuron.leakage_factor = LF
        neuron.leakage_period = LP
        neuron.theta = thetas[1+i]
        neuron.activation_function = IDENTITY
        neuron.membrane_should_reset = False
        neuron.set_supervised_stdp(A, tau, clk_freq, max_weight, min_weight, ground_truths[1 + i])
        network.add_layer(SCTNLayer([neuron]))

    # feedback
    network.connect_by_id(4, 1)
    return network

def simple_resonator(
        freq0,
        lf,
        thetas,
        weights,
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

def flat_weights(resonator):
    ws = []
    for neuron in resonator.neurons[1:]:
        for w in neuron.synapses_weights:
            ws.append(abs(float(f'{w:.3f}')))
    return np.array(ws)

def flat_thetas(resonator):
    return [float(f'{neuron.theta:.3f}') for neuron in resonator.neurons[1:]]


def search_for_parameters(freq0, lf, thetas, weights, phase):
    duration = (2+phase) / freq0

    x = np.linspace(0, duration, int(duration * clk_freq))
    t = x * 2 * np.pi * freq0
    sine_wave = np.sin(t)
    wave_length = int(clk_freq/freq0)

    spikes_window = 500
    resonator = SpikingNetwork(clk_freq)
    resonator.add_amplitude(1000)

    # Encode to pdm
    neuron = create_SCTN()
    neuron.activation_function = IDENTITY
    resonator.add_layer(SCTNLayer([neuron]))
    resonator.log_out_spikes(-1)
    resonator.input_full_data(sine_wave)

    # generate gt
    resonator_input = neuron_output(resonator.neurons[0], freq0, shift_degrees=0, phase_number=phase)

    ground_truth = []
    for phase_shift in [45, 90, 135, 180]:
        phase_shift /= 360
        resonator.input_full_data(sine_wave[int((1-phase_shift)*wave_length):int((20-phase_shift)*wave_length)])
        resonator.log_out_spikes(-1)
        resonator.forget_logs()

        resonator.input_full_data(10*sine_wave[int((1-phase_shift)*wave_length):])
        ground_truth.append(neuron_output(resonator.neurons[0], freq0, phase_number=phase))

    rolling_gt = []
    for i, gt in enumerate(ground_truth):
        rolling_gt.append(events_to_spikes(gt-resonator_input[0], run_window=spikes_window, spikes_arr_size=int(clk_freq/freq0)+1))

    # create a learning_resonator
    resonator = learning_resonator(
        lf=lf,
        freq0=freq0,
        thetas=thetas,
        weights=weights,
        ground_truths=ground_truth,
        A=1e-4,
        time_to_learn=5e-6,
        max_weight=np.inf,
        min_weight=-np.inf,
    )
    learning_rules = [neuron.supervised_stdp for neuron in resonator.neurons[1:]]
    for i in range(len(resonator.neurons)):
        resonator.log_out_spikes(i)

    # start training
    epochs = 1000

    weights_arr = np.zeros((epochs+1, 5))
    weights_arr[0, :] = flat_weights(resonator)

    thetas_arr = np.zeros((epochs+1, 4))
    thetas_arr[0, :] = flat_thetas(resonator)

    mses = np.ones((epochs+1, 4)) * np.inf
    mses[0, :] = np.array([(gt**2).mean() for gt in ground_truth])

    min_mse_th = start_min_mse_th
    epoch_start = 1
    while True:
        max_theta = -0.75
        with tqdm(total=epochs) as pbar:
            for i in range(epoch_start, epochs+epoch_start):
                resonator.input_full_data(sine_wave)
                output = [events_to_spikes(neuron_output(neuron, freq0, phase_number=phase)-resonator_input[0],
                                           run_window=spikes_window,
                                           spikes_arr_size=int(clk_freq/freq0)+1)
                          for neuron in resonator.neurons[1:]]

                weights_arr[i, :] = flat_weights(resonator)

                thetas_arr[i, :] = flat_thetas(resonator)
                thetas_shift = [-(((2*np.mean(o) - spikes_window)/spikes_window)**2)*np.sign(np.mean(o)-spikes_window/2) for o in output]
                for bs, neuron in zip(thetas_shift, resonator.neurons[1:]):
                    neuron.theta += bs
                    if neuron.theta > max_theta:
                        neuron.theta = max_theta

                mses[i, :] = [((gt - o)**2).mean() for gt, o in zip(rolling_gt, output)]

                # activate weights learning
                wave_amplitudes = [o.max() - o.min() for o in output]
                for j, o in enumerate(output):
                    dc = o.mean()
                    neuron = resonator.neurons[1+j]
                    if abs(dc - spikes_window/2) < 10 or neuron.theta <= max_theta:
                        wave_amplitude = output[j].max() - output[j].min()
                        gt_wave_amplitude = rolling_gt[j].max() - rolling_gt[j].min()
                        wave_amplitude_ratio = abs((wave_amplitude - gt_wave_amplitude)/gt_wave_amplitude)
                        neuron.supervised_stdp = learning_rules[j]
                        neuron.supervised_stdp.A = wave_amplitude_ratio * 5e-3
                    else:
                        neuron.supervised_stdp = None
                pbar.set_postfix({'weights': flat_weights(resonator).tolist(), 'thetas': flat_thetas(resonator), 'mse': mses[i, :].mean(),
                                  'amplitudes': wave_amplitudes, 'dc': [int(o.mean()) for o in output]})

                resonator.forget_logs()
                pbar.update(1)

        arg_min_mse = np.sum(mses, axis=1).argmin()
        mean_min_mse = mses[arg_min_mse].mean()
        if mean_min_mse <= min_mse_th:
            print(f'min mse {mean_min_mse}')
            break
        else:
            min_mse_th += 100
            weights_arr = np.concatenate([weights_arr, np.zeros((epochs, 5))])
            thetas_arr = np.concatenate([thetas_arr, np.zeros((epochs, 4))])
            mses = np.concatenate([mses, np.ones((epochs, 4)) * np.inf])
            epoch_start += epochs

    chosen_thetas = thetas_arr[arg_min_mse]
    chosen_weights = weights_arr[arg_min_mse]

    # plot the output of the neurons.
    res_resonator = simple_resonator(
        freq0=freq0,
        lf=lf,
        thetas=chosen_thetas,
        weights=chosen_weights,
    )

    for i in range(len(res_resonator.neurons)):
        res_resonator.log_out_spikes(i)
    res_resonator.input_full_data(sine_wave)

    output = [events_to_spikes(neuron_output(neuron, freq0, phase_number=phase)-resonator_input[0], run_window=spikes_window, spikes_arr_size=int(clk_freq/freq0)+3) for neuron in res_resonator.neurons]

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f'{mses[arg_min_mse].mean()}')
    for i, o in enumerate(output):
        axs[0].plot(o, label=f'output neuron {i}')
    axs[0].legend()

    axs[1].plot(events_to_spikes(resonator_input - resonator_input[0], run_window=spikes_window), label=f'input')
    for i, gt in enumerate(rolling_gt):
        axs[1].plot(gt, label=f'gt {i + 1}')

    axs[1].legend()
    plt.savefig(f'../filters3/clk_{clk_freq}/figures/wave_{int(freq0)}.png', bbox_inches='tight')
    plt.close()

    # run a chirp test
    res_resonator = simple_resonator(
        freq0=freq0,
        lf=lf,
        thetas=chosen_thetas,
        weights=chosen_weights,
    )

    for nid in range(1,5):
        res_resonator.log_out_spikes(nid)

    start_freq = 0
    spectrum = 2 * freq0
    res_resonator.forget_logs()

    step = step_ratio / clk_freq
    test_size = int(spectrum / step)
    test_resonator_on_chirp(
        res_resonator,
        start_freq=start_freq,
        step=step,
        test_size=test_size,
        clk_freq=clk_freq
    )

    snrs = [0] * 4
    peaks = [0] * 4
    best_lp = lp_by_lf(lf, freq0, clk_freq)
    f_resonator = freq_of_resonator(clk_freq, 5, best_lp)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'spikes in window of {spikes_window} freq: {f_resonator:.2f}')
    for nid in range(4):
        spikes_neuron = res_resonator.neurons[nid + 1]
        y_events = spikes_neuron.out_spikes()
        y_spikes = np.zeros(test_size)
        y_spikes[y_events] = 1
        y_spikes = np.convolve(y_spikes, np.ones(spikes_window, dtype=int), 'valid')
        x = np.linspace(start_freq, start_freq+spectrum, len(y_spikes))

        ax = axs[nid//2, nid%2]
        ax.set_ylim(spikes_window/4, 3*spikes_window/4)
        ax.plot(x, y_spikes)
        snr_spikes = y_spikes[3000:]
        snrs[nid] = (snr_spikes.max() - snr_spikes.min())/snr_spikes.std()
        peaks[nid] = x[3000+np.argmax(snr_spikes)]
        ax.set_title(f'neuron id {nid + 1} peak: {peaks[nid]:.3f} snr: {snrs[nid]:.3f}')
    fig.tight_layout()
    plt.savefig(f'../filters3/clk_{clk_freq}/figures/chirp_{int(freq0)}.png', bbox_inches='tight')
    plt.close()

    nid = np.argmax(np.array(snrs) / np.abs(np.array(peaks) - freq0))
    print(f'# {freq0} ~ peak {peaks[nid]:.3f} nid {nid+1} snr {snrs[nid]:.3f}')
    print(f'# chosen_bias={list(chosen_thetas)}')
    print(f'# chosen_weights={list(chosen_weights)}')

    with open(f"../filters3/clk_{clk_freq}/parameters/f_{int(freq0)}.json", 'w') as best_params_f:
        parameters = {
            'freq0': float(freq),
            'f_resonator': float(f_resonator),
            'mse': list(mses[arg_min_mse]),
            'mean_mse': float(mses[arg_min_mse].mean()),
            'thetas': list(chosen_thetas),
            'weights': list(chosen_weights),
            'peaks': [f'{peak:.2f}' for peak in peaks],
            'snrs': [f'{snr:.2f}' for snr in snrs],
            'best_neuron': int(nid + 1)
        }
        json.dump(parameters, best_params_f, indent=4)
    return chosen_thetas, chosen_weights


clk_freq = 1536000
Path(f"../filters3/clk_{clk_freq}/figures").mkdir(parents=True, exist_ok=True)
Path(f"../filters3/clk_{clk_freq}/parameters").mkdir(parents=True, exist_ok=True)

lf = 4
step_ratio = 10
start_min_mse_th = 100
freqs = [104 * (.87 ** i) for i in range(8, 35)]

# lf = 6
# step_ratio = 100
# freqs = [104 * (1.07 ** i) for i in range(38, 60)]

# init_thetas = [-1, -5, -5, -5]
# init_weights = [11, 9, 10, 10, 10]

init_thetas=[ -16.406, -60.501, -41.111, -11.995 ]
init_weights=[ 161.201, 126.148, 120.231, 82.475, 25.93 ]
for freq in freqs:
    init_thetas, init_weights = search_for_parameters(freq, lf, init_thetas, init_weights, phase=20)
