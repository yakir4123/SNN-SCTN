import os

import numpy as np

from tqdm import tqdm
from itertools import product
from helpers.graphs import plot_network
from helpers import timing, load_audio_data, printable_weights
from scripts.rwcp_resonators import snn_based_resonator_for_learning, snn_based_resonator_for_test
from snn.spiking_network import get_labels


def test_input(network, data):
    labels = get_labels(network)
    classes = network.input_full_data(data)
    choose_labeled = np.argmax(classes)
    print(f'Find class {labels[choose_labeled]}. out spikes {labels} = {classes}')
    return labels[choose_labeled]


@timing
def make_neuron_learn(network, audio_label, signals, epochs, continue_learning):
    neuron = network.neurons[-1]
    weight_generations_buffer = np.zeros((epochs * len(signals) + 1, *neuron.synapses_weights.shape))
    weight_generations_buffer[0, :] = neuron.synapses_weights
    if continue_learning:
        previous_learning = np.load(f'neurons_weights/{audio_label}_synapses_weights_generations.npz')
        previous_learning = previous_learning['synapses_weights']
        previous_learning = previous_learning[~np.all(previous_learning == 0, axis=1)]
        weight_generations_buffer[:len(previous_learning), :] = previous_learning

    with tqdm(total=epochs * len(signals)) as pbar:
        for epoch in range(epochs):
            audio_file_indices = np.random.permutation(len(signals))
            for signal_index in range(len(signals)):
                i = epoch * len(signals) + signal_index
                signal = signals[audio_file_indices[signal_index]]

                try:
                    audio_path = f"../datasets/RWCP/{audio_label}/{signal}"
                    data = load_audio_data(audio_path, clk_freq,
                                           resample_time_ms=0,
                                           remove_silence=True)
                except FileNotFoundError:
                    continue

                classes = network.input_full_data(data)
                network.reset_learning()
                weight_generations_buffer[i + 1, :] = neuron.synapses_weights
                weight_generations = weight_generations_buffer[:i + 2, :]

                np.savez_compressed(f'neurons_weights/{audio_label}_synapses_weights_generations.npz',
                                    synapses_weights=weight_generations)
                l1 = np.sum(np.abs(weight_generations_buffer[i + 1, :] - weight_generations_buffer[i, :]))
                pbar.set_description(f"l1 {l1:.2f} |{printable_weights(neuron.synapses_weights)}| spikes {np.sum(classes)}")
                pbar.update()

            # check if didn't learn much from this epoch
            l1 = np.sum(np.abs(weight_generations_buffer[i + 1, :] - weight_generations_buffer[i + 1 - len(signals), :]))
            if epoch > 0 and l1 < .05:
                return weight_generations_buffer
            print(f'\n{neuron.synapses_weights}')
            neuron.set_stdp_ltp(neuron.stdp.A_LTP * 0.95)
            neuron.set_stdp_ltd(neuron.stdp.A_LTD * 0.95)
    return weight_generations_buffer


def learn_neurons(freqs, label, clk_freq, continue_learning=False):
    network = snn_based_resonator_for_learning(freqs, clk_freq)
    # plot_network(network)

    learning_neuron = network.neurons[-1]
    network.log_out_spikes(-1)
    for n in network.layers_neurons[-2].neurons:
        network.log_out_spikes(n._id)

    seed = sum(map(ord, label))
    train_signals, _ = train_test_files(label, train_test_split=.8, seed=seed)
    prev_weights = np.copy(learning_neuron.synapses_weights)
    print(f'pre - learning - {label}\r\n{prev_weights}')
    make_neuron_learn(
        network,
        label,
        signals=train_signals,
        epochs=20,
        continue_learning=continue_learning
    )
    print(f'post - learning - {label}\r\n{learning_neuron.synapses_weights.tolist()}')
    print(
        f'diff - {label}\r\n{dict(zip(freqs, (learning_neuron.synapses_weights - prev_weights).astype(np.int16)))}')

    print('Done.')


def test_neurons(freqs, label, clk_freq):
    print(f'Classify test on {label}')
    network = snn_based_resonator_for_test(freqs, clk_freq)
    # plot_network(network)
    success = 0
    count_test = 0

    seed = sum(map(ord, label))
    _, test_signals = train_test_files(label, train_test_split=.8, seed=seed)

    for file_name in test_signals:
        try:
            audio_path = f"../datasets/RWCP/{label}/{file_name}"
            data = load_audio_data(audio_path, clk_freq,
                                   resample_time_ms=0,
                                   remove_silence=True)
        except FileNotFoundError:
            continue

        is_success = test_input(network, data) == label
        success += is_success
        count_test += 1

    print(f'success rate on {label} is {success / count_test * 100}')


def train_test_files(label, train_test_split=.5, seed=42):
    files_names = np.array(os.listdir(f"../datasets/RWCP/{label}"))

    np.random.seed(seed)
    shuffle = np.random.permutation(len(files_names))
    files_names = files_names[shuffle]
    train = files_names[:int(len(files_names) * train_test_split)]
    test = files_names[int(len(files_names) * train_test_split):]
    return train, test


if __name__ == '__main__':

    clk_freq = int(1.536 * (10 ** 6) * 2)
    # freqs = [
    #     200, 236, 278, 328, 387, 457,
    #     751, 887, 1046, 1235, 1457,
    #     1719, 2029, 2825, 3934, 5478
    # ]

    freqs = [
        751, 1046, 3934
    ]
    # learn_neurons(freqs, 'bottle1', clk_freq)
    # learn_neurons(freqs, 'buzzer', clk_freq)
    # learn_neurons(freqs, 'phone4', clk_freq)

    # test_neurons(freqs, 'bottle1', clk_freq)
    # test_neurons(freqs, 'buzzer', clk_freq)
    test_neurons(freqs, 'phone4', clk_freq)
