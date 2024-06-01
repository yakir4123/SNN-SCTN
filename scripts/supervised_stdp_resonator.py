import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm import tqdm
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import Polynomial


from snn.layers import SCTNLayer
from snn.resonator import lp_by_lf, trained_resonator
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import create_SCTN, IDENTITY
from snn.resonator import test_resonator_on_chirp, freq_of_resonator


def neuron_output(neuron, signal_freq, shift_degrees=0, phase_number=20):
    y_events = neuron.out_spikes()
    samples_per_cycle = clk_freq / signal_freq
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


def argmax(arr):
    return .5 * (np.argmax(arr) + len(arr) - np.argmax(arr[::-1]) - 1)

def read_or_default(freq0, default_thetas, default_weights, lf=4, clk_freq=1536000):
    try:
        with open(f'../filters{lf}{postfix}/clk_{clk_freq}/parameters/f_{int(freq0)}.json') as f:
            full_parameters = json.load(f)
            return full_parameters['thetas'], full_parameters['weights']
    except FileNotFoundError:
        return default_thetas, default_weights
def estimated_parameters_for_resonator(freq0, lf, clk_freq, polynom_deg=4):
    filters_dir = f'../filters{lf}/clk_{clk_freq}/parameters/'

    resonators_parameters = []
    for f in os.listdir(filters_dir):
        with open(f'{filters_dir}/{f}') as f:
            hp = {}
            full_parameters = json.load(f)
            f_resonator = float(full_parameters['f_resonator'])
            # if full_parameters['lf'] != 5 or int(full_parameters['freq0']//1) not in chosen_freqs:
            #     continue
            hp['f_resonator'] = f_resonator  # float(full_parameters['peaks'][2])
            for i in range(4):
                hp[f'th{i}'] = full_parameters['thetas'][i]
            for i in range(5):
                hp[f'w{i}'] = full_parameters['weights'][i]
            resonators_parameters.append(hp)

    df = pd.DataFrame(resonators_parameters)
    df = df.set_index('f_resonator').sort_index()

    for col in df.columns:
        fit = Polynomial.fit(df.index, df[col], deg=polynom_deg)
        df[f'{col}-fit'] = fit(df.index)
    poly_th0 = Polynomial.fit(df.index, df['th0'], deg=polynom_deg)
    poly_th1 = Polynomial.fit(df.index, df['th1'], deg=polynom_deg)
    poly_th2 = Polynomial.fit(df.index, df['th2'], deg=polynom_deg)
    poly_th3 = Polynomial.fit(df.index, df['th3'], deg=polynom_deg)

    poly_w0 = Polynomial.fit(df.index, df['w0'], deg=polynom_deg)
    poly_w1 = Polynomial.fit(df.index, df['w1'], deg=polynom_deg)
    poly_w2 = Polynomial.fit(df.index, df['w2'], deg=polynom_deg)
    poly_w3 = Polynomial.fit(df.index, df['w3'], deg=polynom_deg)
    poly_w4 = Polynomial.fit(df.index, df['w4'], deg=polynom_deg)

    thetas = [poly_th0(freq0), poly_th1(freq0), poly_th2(freq0), poly_th3(freq0)]
    weights = [poly_w0(freq0), poly_w1(freq0), poly_w2(freq0), poly_w3(freq0), poly_w4(freq0)]
    return thetas, weights


def search_for_parameters(freq0, lf, thetas, weights, phase, xi=22.5, learnable_neurons=None, gain=8):
    learnable_neurons = learnable_neurons or [0, 1, 2, 3]
    best_lp = lp_by_lf(lf, freq0, clk_freq)
    freq0 = freq_of_resonator(clk_freq, lf, best_lp)

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
    rresonator_input = events_to_spikes(resonator_input - resonator_input[0], run_window=spikes_window,
                                        spikes_arr_size=int(clk_freq / freq0) + 1)
    while True:
        ground_truth = []
        desired_xi = np.array([xi, xi + 45, xi+90, xi + 135])
        for phase_shift in desired_xi:
            # for mse purpose
            phase_shift /= 360
            resonator.input_full_data(sine_wave[int((1-phase_shift)*wave_length):int((20-phase_shift)*wave_length)])
            resonator.log_out_spikes(-1)
            resonator.forget_logs()

            resonator.input_full_data(gain*sine_wave[int((1-phase_shift)*wave_length):])
            ground_truth.append(neuron_output(resonator.neurons[0], freq0, phase_number=phase))


        rolling_gt = []
        for i, gt in enumerate(ground_truth):
            rolling_gt.append(events_to_spikes(gt-resonator_input[0], run_window=spikes_window, spikes_arr_size=int(clk_freq/freq0)+1))

        gt_wave_amplitudes = [(o.max(), o.min()) for o in rolling_gt]
        if gt_wave_amplitudes[0][0] > 415:
            gain *= .97
        elif gt_wave_amplitudes[0][0] < 410:
            gain *= 1.03
        else:
            print(f' gain {gain} ({gt_wave_amplitudes[0]})')
            break


    # create a learning_resonator
    resonator = learning_resonator(
        lf=lf,
        freq0=freq0,
        thetas=thetas,
        weights=weights,
        ground_truths=ground_truth,
        A=10e-5,
        time_to_learn=1e-5,
        max_weight=np.inf,
        min_weight=-np.inf,
    )
    learning_rules = [neuron.supervised_stdp for neuron in resonator.neurons[1:]]
    for i, neuron in enumerate(resonator.neurons):
        resonator.log_out_spikes(i)
        neuron.supervised_stdp = None

    min_mse_weights = flat_weights(resonator)
    min_mse_thetas = flat_thetas(resonator)
    min_mse = np.array([(gt**2).mean() for gt in ground_truth]).mean()*100

    min_mse_weights_tuned = flat_weights(resonator)
    min_mse_thetas_tuned = flat_thetas(resonator)
    min_mse_tuned = np.array([(gt**2).mean() for gt in ground_truth]).mean()*100

    momentum = [0] * 4
    max_theta = -.75

    y_epsilon = spikes_window * 0.035
    x_epsilon = len(rresonator_input) * 10 / 360
    gt_peaks = [argmax(gt) for gt in rolling_gt]

    count_to_finish = -1
    epochs_after_tuning = 100
    epoch = 0
    with tqdm() as pbar:
        while count_to_finish != 0:
            learn = True
            count_to_finish -= 1
            tuned_parameters = 0
            for neuron in resonator.neurons:
                neuron.membrane_potential = 0
                neuron.log_rand_gauss_var = 0
            resonator.forget_logs()
            resonator.input_full_data(sine_wave)
            output = [events_to_spikes(neuron_output(neuron, freq0, phase_number=phase)-resonator_input[0],
                                       run_window=spikes_window,
                                       spikes_arr_size=int(clk_freq/freq0)+1)
                      for neuron in resonator.neurons[1:]]

            for j, o in enumerate(output):
                o_max, o_min = o.max(), o.min()
                o_argmax = argmax(o)
                if abs(o_argmax - gt_peaks[j]) <= x_epsilon:
                    tuned_parameters += 1
                # if ((o_max - gt_wave_amplitudes[j][0]) >= -y_epsilon and
                #         (o_min - gt_wave_amplitudes[j][1]) <= y_epsilon
                # ):
                if (abs(o_max - gt_wave_amplitudes[j][0]) <= y_epsilon and
                        abs(o_min - gt_wave_amplitudes[j][1]) <= y_epsilon
                ):
                    tuned_parameters += 1
            if tuned_parameters == 8 and count_to_finish < 0:
                count_to_finish = epochs_after_tuning

            mses = np.array([((gt - o)**2).mean() for gt, o in zip(rolling_gt, output)])
            curr_mse = mses.mean()
            if curr_mse < min_mse:
                if curr_mse < 15:
                    count_to_finish = epochs_after_tuning
                    if min_mse_tuned > 3 * curr_mse:
                        min_mse_thetas_tuned = flat_thetas(resonator)
                        min_mse_weights_tuned = flat_weights(resonator)

                min_mse = curr_mse
                min_neurons_mses = mses
                learn = False
                min_mse_thetas = flat_thetas(resonator)
                min_mse_weights = flat_weights(resonator)
            if tuned_parameters == 8 and curr_mse < min_mse_tuned:
                min_mse_tuned = curr_mse
                min_mse_thetas_tuned = flat_thetas(resonator)
                min_mse_weights_tuned = flat_weights(resonator)

            if learn:
                thetas_shift = [-.2*(((2*np.mean(o) - spikes_window)/spikes_window)**2)*np.sign(np.mean(o)-spikes_window/2) for o in output]
                for j, neuron in enumerate(resonator.neurons[1:]):
                    bs = thetas_shift[j]
                    momentum[j] = bs + momentum_beta * momentum[j]
                    neuron.theta += momentum[j]
                    if neuron.theta > max_theta:
                        neuron.theta = max_theta

            # activate weights learning
            peaks = [argmax(o)for o in output]
            for j, o in enumerate(output):
                if j not in learnable_neurons:
                    continue
                o_max = o.max()
                o_min = o.min()
                neuron = resonator.neurons[1 + j]
                o_argmax = argmax(o)

                if (learn and
                        abs(o_argmax - gt_peaks[j]) <= 5 * x_epsilon# and
                        # abs(o_max - gt_wave_amplitudes[j][0]) > y_epsilon/4 and
                        # abs(o_min - gt_wave_amplitudes[j][1]) > y_epsilon/4
                ):
                    # 100 mse -> stretch_or_shrink_scale 0.001
                    stretch_or_shrink_scale = 2*(mses[j]*1000//1e4) / 1e4
                    if gt_wave_amplitudes[j][1] < o_min < o_max < gt_wave_amplitudes[j][0]:
                        neuron.theta -= stretch_or_shrink_scale
                        neuron.synapses_weights[0] += 2 * stretch_or_shrink_scale #/ len(neuron.synapses_weights)
                        if j == 0:
                            neuron.synapses_weights[1] -= 2 * stretch_or_shrink_scale  # / len(neuron.synapses_weights)
                    elif o_min < gt_wave_amplitudes[j][1] < gt_wave_amplitudes[j][0] < o_max:
                        stretch_or_shrink_scale *= 2
                        neuron.theta += stretch_or_shrink_scale
                        neuron.synapses_weights[0] -= 2 * stretch_or_shrink_scale #/ len(neuron.synapses_weights)
                        if j == 0:
                            neuron.synapses_weights[1] += 2 * stretch_or_shrink_scale  # / len(neuron.synapses_weights)

                if not learn:
                    neuron.supervised_stdp = None
                else:
                    wave_amplitude = o_max - o_min
                    gt_wave_amplitude = gt_wave_amplitudes[j][0] - gt_wave_amplitudes[j][1]
                    # number between 0 - 1 represent [0 - gt_wave_amplitude]
                    wave_amplitude_ratio = abs((wave_amplitude - gt_wave_amplitude) / gt_wave_amplitude)
                    # number between 0 - 1 represent [0 - 180]
                    phase_diff_ratio = min(abs(peaks[j] - gt_peaks[j]) / len(o), 10/180)
                    neuron.supervised_stdp = learning_rules[j]
                    neuron.supervised_stdp.A = (1 + wave_amplitude_ratio) * 10e-5
                    neuron.supervised_stdp.tau = 1e-5 * clk_freq / 2 * (1 + phase_diff_ratio)

            # check for xi for all of the neurons
            x = np.linspace(0, 360, len(rresonator_input))
            x0 = int(argmax(rresonator_input))
            xi_s = np.array([x[int(argmax(output[i]))] - x[x0] for i in range(4)])

            wave_amplitudes = np.array([o.max() - o.min() for o in output])
            pbar.set_postfix({'weights': flat_weights(resonator).tolist(),
                              'thetas': flat_thetas(resonator), 'mse': curr_mse,
                              'amplitudes': wave_amplitudes.tolist(), 'dc': [int(o.mean()) for o in output],
                              'min_weight': min_mse_weights.tolist(), 'min_thetas': min_mse_thetas, 'min_mse': min_mse,
                              'min_weight_tuned': min_mse_weights_tuned, 'min_thetas_tuned': min_mse_thetas_tuned, 'min_mse_tuned': min_mse_tuned,
                              'tuned_parameters': tuned_parameters, 'xi': xi_s.tolist(),
                              'epochs_left': epochs_after_tuning if count_to_finish < 0 else count_to_finish})

            xi_diff = np.abs(xi_s - desired_xi)
            ampl_diff = np.abs(wave_amplitudes - 130)
            resonator.forget_logs()
            pbar.update(1)
            epoch += 1
            # if all(xi_diff < 5) and all(ampl_diff < 10):
            #     break

            # if epoch == 10000:
            #     min_mse_thetas_tuned = min_mse_thetas
            #     min_mse_weights_tuned = min_mse_weights
            #     break


    # chosen_thetas = flat_thetas(resonator)
    # chosen_weights = flat_weights(resonator)
    chosen_thetas = min_mse_thetas_tuned
    chosen_weights = min_mse_weights_tuned

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
    fig.suptitle(f'{min_mse}')
    for i, o in enumerate(output):
        axs[0].plot(o, label=f'output neuron {i}')
    axs[0].legend()

    axs[1].plot(events_to_spikes(resonator_input - resonator_input[0], run_window=spikes_window), label=f'input')
    for i, gt in enumerate(rolling_gt):
        axs[1].plot(gt, label=f'gt {i + 1}')

    axs[1].legend()
    plt.savefig(f'../filters{lf}{postfix}/clk_{clk_freq}/figures/wave_{int(freq0)}.png', bbox_inches='tight')
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

    step = 150 / clk_freq
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
    f_resonator = freq_of_resonator(clk_freq, lf, best_lp)
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
    plt.savefig(f'../filters{lf}{postfix}/clk_{clk_freq}/figures/chirp_{int(freq0)}.png', bbox_inches='tight')
    plt.close()

    nid = np.argmax(np.array(snrs) / np.abs(np.array(peaks) - freq0))
    print(f'# {freq0} ~ peak {peaks[nid]:.3f} nid {nid+1} snr {snrs[nid]:.3f}')
    print(f'# chosen_bias={list(chosen_thetas)}')
    print(f'# chosen_weights={list(chosen_weights)}')

    with open(f"../filters{lf}{postfix}/clk_{clk_freq}/parameters/f_{int(freq0)}.json", 'w') as best_params_f:
        parameters = {
            'freq0': float(freq),
            'f_resonator': float(f_resonator),
            "lf": lf,
            'mse': list(min_neurons_mses),
            'mean_mse': float(min_mse),
            'thetas': list(chosen_thetas),
            'weights': list(chosen_weights),
            'peaks': [f'{peak:.2f}' for peak in peaks],
            'snrs': [f'{snr:.2f}' for snr in snrs],
            'best_neuron': int(nid + 1)
        }
        json.dump(parameters, best_params_f, indent=4)
    return chosen_thetas, chosen_weights


def run_chirp(freq0, spikes_window=500):
# run a chirp test
    res_resonator = trained_resonator(
        freq0=freq0,
        filters_folder='filters4_xi0'
    )

    for nid in range(1,5):
        res_resonator.log_out_spikes(nid)

    start_freq = 0
    spectrum = 2 * freq0
    res_resonator.forget_logs()

    step = 100 / clk_freq
    test_size = int(spectrum / step)
    test_resonator_on_chirp(
        res_resonator,
        start_freq=start_freq,
        step=step,
        test_size=test_size,
        clk_freq=clk_freq
    )

    spikes_neuron = res_resonator.neurons[-1]
    y_events = spikes_neuron.out_spikes()
    y_spikes = np.zeros(test_size)
    y_spikes[y_events] = 1
    y_spikes = np.convolve(y_spikes, np.ones(spikes_window, dtype=int), 'valid')

    with open(f"../filters{lf}{postfix}/clk_{clk_freq}/chirp/f_{int(freq0)}.json", 'w') as best_params_f:
        parameters = {
            'freq0': float(freq),
            'max': np.max(y_spikes),
            "mean": np.mean(y_spikes),
            'min': np.min(y_spikes),
        }
        json.dump(parameters, best_params_f, indent=4)

lf = 4
clk_freq = 1536000
# postfix = '_std'
postfix = '_xi0'
Path(f"../filters{lf}{postfix}/clk_{clk_freq}/figures").mkdir(parents=True, exist_ok=True)
Path(f"../filters{lf}{postfix}/clk_{clk_freq}/parameters").mkdir(parents=True, exist_ok=True)
Path(f"../filters{lf}{postfix}/clk_{clk_freq}/chirp").mkdir(parents=True, exist_ok=True)

momentum_beta = .0
# 0 - 52
# 100 - 112
des_freqs = list(set([int(freq_of_resonator(clk_freq, lf, lp)) for lp in range(144-97, 144-100, -1)]))
freqs = sorted(des_freqs)

xi0_freqs = os.listdir('../filters4_xi0/clk_1536000/parameters')

freqs = xi0_freqs
freqs = sorted([int(f[2:-5]) for f in freqs])[2 + 40*3::3]


freqs = [105, 110, 115, 128, 130, 159, 160, 166, 190, 195, 221, 250, 288, 305, 339, 347, 372, 402, 412, 436, 462, 477, 509, 526, 545, 587, 636, 694, 763, 898]
# 175, 279, 898, 954
# freqs = [898]
# init_thetas = [-106.507, -93.156, -90.817, -81.602]
# init_weights = [359.011, 151.96, 182.411, 179.858, 167.44]

# init_thetas = [-117.517, -100.7, -96.316, -85.802]
# init_weights = [387.914, 160.052, 197.197, 190.622, 177.059]
# print(f'freqs left {len(freqs)} - {freqs}')
# for i, freq in enumerate(freqs):
#     print(i, freq)
#     # init_thetas, init_weights = read_or_default(freq, default_thetas=init_thetas, default_weights=init_weights)
#     init_thetas, init_weights = search_for_parameters(freq, lf, init_thetas, init_weights,
#                                                       phase=20, xi=0, learnable_neurons=[0, 1, 2, 3],
#                                                       gain=26)

# run chirps on these signals:
freqs = np.array([
    [110, 130, 160, 190, 221, 250, 195, 282, 305, 347, 402, 436],
    [288, 305, 339, 372, 412, 462, 898, 105, 115, 128, 159, 166],
    [509, 545, 587, 636, 694, 763, 477, 526,   0,   0,   0,   0]
])[2, :]

print(f'run on {freqs}')
for freq in freqs:
    if freq > 0:
        run_chirp(freq, spikes_window=100)


