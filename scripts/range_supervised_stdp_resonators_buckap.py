

import json
import numpy as np
from tqdm import tqdm
from snn.layers import SCTNLayer
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import create_SCTN, IDENTITY
from snn.resonator import freq_of_resonator, lp_by_lf

# Global Variables
JSON_FILE_PATH = "../filters4_xi0/clk_1536000/parameters/ecg/"


# ========================================================================================
def neuron_output(neuron, signal_freq, shift_degrees=0, phase_number=10):
    y_events = neuron.out_spikes()
    samples_per_cycle = clk_freq / signal_freq
    samples_per_degree = samples_per_cycle / 360
    shift_samples = int(shift_degrees * samples_per_degree)
    y_events = y_events[
        (y_events > ((phase_number - 1) / signal_freq) * clk_freq + shift_samples) &
        (y_events < ((phase_number / signal_freq) * clk_freq) + shift_samples)
        ]
    if len(y_events) == 0:
        return y_events
    return y_events


# ========================================================================================
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


# ========================================================================================
# def amplify_spikes(base_spikes, spikes_window, base_min, base_max, amplify):
#     base_spikes = events_to_spikes(base_spikes - resonator_input[0], spikes_arr_size=int(clk_freq / freq0) + 1)
#     result = np.zeros_like(base_spikes)
#     result[np.arange(len(base_spikes)) % 2 == 1] = 1
#     dc = (base_max + base_min) / 2
#     for i in range(spikes_window, len(result), spikes_window):
#         slice_spikes = base_spikes[i:i + spikes_window]
#         if len(slice_spikes) < spikes_window:
#             break
#         curr_spikes = np.sum(slice_spikes) - dc
#         new_spikes = int(curr_spikes * amplify)
#
#         if new_spikes > 0:
#             spikes_ts = np.linspace(0, spikes_window, new_spikes).astype(int) // 2 * 2 + i
#             result[spikes_ts] = 1
#         else:
#             spikes_ts = np.linspace(0, spikes_window, abs(new_spikes)).astype(int) // 2 * 2 - 1 + i
#             result[spikes_ts] = 0
#     result = np.where(result == 1)[0]
#     return result


# ========================================================================================
def smooth(arr, size):
    filter = np.exp(-(np.arange(size) - size // 2) ** 2 / (2 * (size / 4) ** 2))
    normalized_filter = filter / np.sum(filter)
    res = np.convolve(arr, normalized_filter, 'same')
    res[:size] = arr[:size]
    res[-size:] = arr[-size:]
    return res


# ========================================================================================
def simple_resonator(
        freq0,
        clk_freq,
        lf,
        thetas,
        weights,
):
    LF = lf
    LP = lp_by_lf(LF, freq0, clk_freq)
    network = SpikingNetwork()
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
        neuron.synapses_weights = np.array([weights[2 + i]], dtype=np.float64)
        neuron.leakage_factor = LF
        neuron.leakage_period = LP
        neuron.theta = thetas[1 + i]
        neuron.activation_function = IDENTITY
        neuron.membrane_should_reset = False
        network.add_layer(SCTNLayer([neuron]))

    # feedback
    network.connect_by_id(4, 1)
    return network


# ========================================================================================
def learning_resonator(
        freq0,
        clk_freq,
        lf,
        thetas,
        weights,
        ground_truths,
        A,
        time_to_learn,
        max_weight,
        min_weight,
):
    network = simple_resonator(freq0, clk_freq, lf, thetas, weights)
    tau = clk_freq * time_to_learn / 2
    network.neurons[1].set_supervised_stdp(A, tau, clk_freq, max_weight, min_weight, ground_truths[0])
    for i in range(3):
        network.neurons[2 + i].set_supervised_stdp(A, tau, clk_freq, max_weight, min_weight, ground_truths[1 + i])
    return network


# ========================================================================================
def flat_weights(resonator):
    ws = []
    for neuron in resonator.neurons[1:]:
        for w in neuron.synapses_weights:
            ws.append(abs(float(f'{w:.3f}')))
    return np.array(ws)


# ========================================================================================
def flat_thetas(resonator):
    return [float(f'{neuron.theta:.3f}') for neuron in resonator.neurons[1:]]


# ========================================================================================
def argmax(arr):
    return .5 * (np.argmax(arr) + len(arr) - np.argmax(arr[::-1]) - 1)


# ========================================================================================
def leraning_algorithm():
    epochs = 650

    heights_ratios = np.ones(6)
    heights_ratios[0] = 2

    weights = np.zeros((epochs + 1, 5))
    weights[0, :] = flat_weights(resonator)

    biases = np.zeros((epochs + 1, 4))
    biases[0, :] = flat_thetas(resonator)

    mses = np.ones((epochs + 1, 5)) * np.inf
    mses[0, :4] = np.array([(gt ** 2).mean() for gt in ground_truth])
    mses[0, 4] = mses[0, :4].mean()
    min_mse = mses[0, 4]

    stdp_amplitude = np.ones((epochs + 1, 4)) * np.inf
    stdp_amplitude[0, :] = np.array([lr.A for lr in learning_rules])
    amplitude_ratio = np.zeros((epochs + 1, 4))

    stdp_tau = np.ones((epochs + 1, 4)) * np.inf
    stdp_tau[0, :] = np.array([lr.tau for lr in learning_rules])

    phase_ratio = np.zeros((epochs + 1, 4))

    y_epsilon = spikes_window * 0.036
    x_epsilon = len(rresonator_input) * 7 / 360

    gt_peaks = [argmax(gt) for gt in rolling_gt]

    areas_sns = [[], [], [], []]
    start_sns = [-1] * 4
    momentum = [0] * 4
    max_theta = -0.75
    tuned_parameters = 0
    count_after_tune = -1
    with tqdm() as pbar:
        i = 1
        while count_after_tune != 0:
            count_after_tune -= 1
            run_with_stdp = True
            tuned_parameters = 0
            for neuron in resonator.neurons:
                neuron.membrane_potential = 0
                neuron.log_rand_gauss_var = 0
            resonator.forget_logs()
            resonator.input_full_data(sine_wave)

            # mse
            output = [events_to_spikes(neuron_output(neuron, freq0) - resonator_input[0],
                                       run_window=spikes_window,
                                       spikes_arr_size=int(clk_freq / freq0) + 1)
                      for neuron in resonator.neurons[1:]]
            # max_y = 0
            mses[i, :4] = [((gt - o) ** 2).mean() for gt, o in zip(rolling_gt, output)]
            mses[i, 4] = mses[i, :4].mean()
            if mses[i, 4] < min_mse:
                min_mse = mses[i, 4]
                run_with_stdp = False

            thetas_shift = [
                -.2 * (((2 * np.mean(o) - spikes_window) / spikes_window) ** 2) * np.sign(
                    np.mean(o) - spikes_window / 2)
                for o in output]
            for j, neuron in enumerate(resonator.neurons[1:]):
                bs = thetas_shift[j]
                momentum[j] = bs + momentum_beta * momentum[j]
                neuron.theta += momentum[j]
                if neuron.theta > max_theta:
                    neuron.theta = max_theta

            weights[i, :] = flat_weights(resonator)
            # delta_weights = weights[1:, :] - weights[:-1, :]
            biases[i, :] = flat_thetas(resonator)
            delta_biases = biases[1:, :] - biases[:-1, :]
            delta_biases[i - 1, :] = 0
            peaks = [argmax(o) for o in output]
            # activate weights learning
            for j, o in enumerate(output):
                # dc = o.mean()
                o_max = o.max()
                o_min = o.min()
                neuron = resonator.neurons[1 + j]
                # first 2 conditions to check if the amplitude is on the right place.
                # next condition is to check if the peak is in the right place.
                o_argmax = argmax(o)

                if (abs(o_argmax - gt_peaks[j]) <= 1.5 * x_epsilon and
                        abs(o_max - gt_wave_amplitudes[j][0]) > y_epsilon / 3 and
                        abs(o_min - gt_wave_amplitudes[j][1]) > y_epsilon / 3
                ):
                    if start_sns[j] == -1:
                        start_sns[j] = i
                    areas_j = areas_sns[j]
                    if areas_j == [] or len(areas_j[-1]) == 2:
                        areas_j.append((i,))
                    stretch_or_shrink_scale = (mses[i, j] * 1000 // 1e3) / 1e3
                    if gt_wave_amplitudes[j][1] < o_min < o_max < gt_wave_amplitudes[j][0]:
                        neuron.theta -= stretch_or_shrink_scale
                        neuron.synapses_weights[0] += 2 * stretch_or_shrink_scale
                        if j == 0:
                            neuron.synapses_weights[1] -= 2 * stretch_or_shrink_scale
                    elif o_min < gt_wave_amplitudes[j][1] < gt_wave_amplitudes[j][0] < o_max:
                        stretch_or_shrink_scale *= 2
                        neuron.theta += stretch_or_shrink_scale
                        neuron.synapses_weights[0] -= 2 * stretch_or_shrink_scale  # / len(neuron.synapses_weights)
                        if j == 0:
                            neuron.synapses_weights[1] += 2 * stretch_or_shrink_scale  # / len(neuron.synapses_weights)
                else:
                    areas_j = areas_sns[j]
                    if areas_j != [] and len(areas_j[-1]) == 1:
                        areas_j[-1] = (areas_j[-1][0], i)

                if abs(o_argmax - gt_peaks[j]) <= x_epsilon:
                    tuned_parameters += 1
                if (abs(o_max - gt_wave_amplitudes[j][0]) <= y_epsilon
                ):
                    tuned_parameters += 1

                if not run_with_stdp:
                    neuron.supervised_stdp = None
                    amplitude_ratio[i, j] = amplitude_ratio[i - 1, j]
                    phase_ratio[i, j] = phase_ratio[i - 1, j]
                else:
                    wave_amplitude = o_max - o_min
                    gt_wave_amplitude = gt_wave_amplitudes[j][0] - gt_wave_amplitudes[j][1]
                    wave_amplitude_ratio = abs((
                                                       wave_amplitude - gt_wave_amplitude) / gt_wave_amplitude)  # number between 0 - 1 represent [0 - gt_wave_amplitude]
                    neuron.supervised_stdp = learning_rules[j]
                    neuron.supervised_stdp.A = (1 + wave_amplitude_ratio) * 10e-5
                    neuron.supervised_stdp.tau = clk_freq / freq0 * (10 / 180 + abs(peaks[j] - gt_peaks[j]) / len(o))

                amplitude_ratio[i, j] = abs((o_max - gt_wave_amplitudes[j][0]))

            wave_amplitudes = [(o.max(), o.min()) for o in output]
            # ================================================
            pbar.set_postfix(
                {'weights': flat_weights(resonator).tolist(), 'thetas': flat_thetas(resonator),
                 'mse': mses[i, :].mean(),
                 'amplitudes': wave_amplitudes, 'dc': [o.mean() for o in output], 'tuned_parameters': tuned_parameters})
            # ================================================

            pbar.update(1)

            i = (i + 1) % epochs
            if i == 0:
                i = 1

            if tuned_parameters == 8 and count_after_tune < 0:
                count_after_tune = 1
                print('tune')
                mse_to_json = list(mses[i - 1])
                print('mse', list(mses[i - 1]))

    # Create a dictionary with the parameters

    params = {
        "clk_freq": clk_freq,
        #"input_freq0": input_freq0,
        "freq0": freq0,
        "lf": lf,
        "lp": best_lp,
        "chosen_bias": chosen_bias,
        "chosen_weights": chosen_weights,
        "mse": mse_to_json,
        "mse_mean": sum(mse_to_json) / len(mse_to_json),
        "weight_results": flat_weights(resonator).tolist(),
        "theta_results": flat_thetas(resonator),
        "iterations": str(i - 1)
    }

    # Construct the file path with freq0 in the name
    json_file_path = JSON_FILE_PATH + "f_" + str(freq0) + ".json"

    # Write the dictionary to the JSON file
    with open(json_file_path, "w") as json_file:
        json.dump(params, json_file, indent=4)

    return flat_weights(resonator).tolist(), flat_thetas(resonator)
    # return {'weights': flat_weights(resonator).tolist(), 'thetas': flat_thetas(resonator), 'mse': mses[i, :].mean(),
    #         'amplitudes': wave_amplitudes, 'dc': [o.mean() for o in output], 'tuned_parameters': tuned_parameters}




# ========================================================================================




if __name__ == '__main__':

    # ========================================================================================
    freq0 =  104
    end_freq = 10
    step_size = -0.4
    chosen_weights = [8.278,
        6.554,
        6.581,
        9.322,
        9.596]
    chosen_bias = [0.867,
        -3.2,
        -4.472,
        -4.604]
    best_lp=76
    clk_freq = 1536000
    #for i in range(start_freq, end_freq - 1, -10):
    #for i in range(start_freq, end_freq):
    #for i in np.arange(start_freq, end_freq, step_size):
    while(0<freq0):
        lf = 5

        freq0 = freq_of_resonator(clk_freq, lf, best_lp)
        print(best_lp)
        print("input_freq0 = ", freq0)
        gain = 12
        duration = 15 / freq0
        x = np.linspace(0, duration, int(duration * clk_freq))
        t = x * 2 * np.pi * freq0
        sine_wave = np.sin(t)
        spikes_window = 500
        wave_length = int(clk_freq / freq0)
        ground_truth = []
        phase_shifts = [0] + [45] * 3
        phase_shifts = np.cumsum(phase_shifts)
        rolling_gt = []
        momentum_beta = .0

        # ========================================================================================

        resonator = SpikingNetwork()
        resonator.add_amplitude(1000)
        # Encode to pdm
        neuron = create_SCTN()
        neuron.activation_function = IDENTITY
        resonator.add_layer(SCTNLayer([neuron]))
        resonator.log_out_spikes(-1)
        resonator.input_full_data(sine_wave)
        resonator_input = neuron_output(resonator.neurons[0], freq0, shift_degrees=0)
        rresonator_input = events_to_spikes(resonator_input - resonator_input[0], run_window=spikes_window,
                                            spikes_arr_size=int(clk_freq / freq0) + 1)

        for phase_shift in phase_shifts:
            phase_shift /= 360
            resonator.input_full_data(
                sine_wave[int((1 - phase_shift) * wave_length):int((20 - phase_shift) * wave_length)])
            resonator.log_out_spikes(-1)
            resonator.forget_logs()

            resonator.input_full_data(gain * sine_wave[int((1 - phase_shift) * wave_length):])
            ground_truth.append(neuron_output(resonator.neurons[0], freq0))
        for i, gt in enumerate(ground_truth):
            rolling_gt.append(
                events_to_spikes(gt - resonator_input[0], run_window=spikes_window,
                                 spikes_arr_size=int(clk_freq / freq0) + 1))

        gt_wave_amplitudes = [(o.max(), o.min()) for o in rolling_gt]
        resonator = learning_resonator(
            freq0=freq0,
            clk_freq=clk_freq,
            lf=lf,
            thetas=chosen_bias,
            weights=chosen_weights,
            ground_truths=ground_truth,
            A=2e-4,
            time_to_learn=1e-5,
            max_weight=np.inf,
            min_weight=-np.inf,
        )
        learning_rules = [neuron.supervised_stdp for neuron in resonator.neurons[1:]]
        for i, neuron in enumerate(resonator.neurons):
            resonator.log_out_spikes(i)
            neuron.supervised_stdp = None

        chosen_weights, chosen_bias = leraning_algorithm()
        best_lp+=1