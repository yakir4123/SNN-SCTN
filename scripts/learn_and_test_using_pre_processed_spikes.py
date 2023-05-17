import os
import time

import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import product
from utils.graphs import plot_network
from utils import timing, load_audio_data, printable_weights
from scripts.rwcp_resonators import snn_based_resonator_for_learning, snn_based_resonator_for_test, labeled_sctn_neuron, \
    learning_neuron, create_neuron_for_labeling
from snn.layers import SCTNLayer
from snn.spiking_network import get_labels, SpikingNetwork


def create_test_network(clk_freq):
    network = SpikingNetwork(clk_freq)
    layer = SCTNLayer([
        labeled_sctn_neuron('bottle1'),
        labeled_sctn_neuron('buzzer'),
        labeled_sctn_neuron('phone4'),
    ])
    network.add_layer(layer)
    return network


def create_learning_network(clk_freq, freqs):
    network = SpikingNetwork(clk_freq)
    layer = SCTNLayer([
        learning_neuron(freqs, clk_freq),
    ])
    network.add_layer(layer)
    return network


def load_spikes_data(label, file_name, freqs):
    spikes = pd.DataFrame \
        .from_dict(dict(
            np.load(f'..\datasets\RWCP_spikes\\{label}\\{file_name}')
        ))
    columns = [f'f{f}' for f in freqs]
    return spikes[columns].to_numpy()


def test_input(network: SpikingNetwork, spikes):
    labels = get_labels(network)
    classes = network.input_full_data_spikes(spikes)

    label_with_first_spike = ''
    argmin_first_spikes = np.inf

    for neuron in network.layers_neurons[-1].neurons:
        neurons_spikes = neuron.out_spikes()
        first_spike = np.argmax(neurons_spikes)
        if first_spike < argmin_first_spikes:
            argmin_first_spikes = first_spike
            label_with_first_spike = neuron.label

    choose_labeled = np.argmax(classes)
    # print(f'Find class {labels[choose_labeled]}. out spikes {labels} = {classes}')
    print(f'Most active {labels[choose_labeled]}, First spike {label_with_first_spike}. out spikes {dict(zip(labels, classes))}')
    return labels[choose_labeled]


@timing
def make_neuron_learn(network: SpikingNetwork, audio_label, signals, epochs):
    neuron = network.neurons[-1]
    network.log_out_spikes(-1)
    network.log_membrane_potential(-1)
    neuron.membrane_sample_max_window = np.zeros(1).astype('float32')

    weight_generations_buffer = np.zeros((epochs * len(signals) + 1, *neuron.synapses_weights.shape))
    weight_generations_buffer[0, :] = neuron.synapses_weights

    compare_epochs_by = signals[0]
    with tqdm(total=epochs * len(signals)) as pbar:
        for epoch in range(epochs):
            audio_file_indices = np.random.permutation(len(signals))
            for signal_index in range(len(signals)):
                i = epoch * len(signals) + signal_index
                signal = signals[audio_file_indices[signal_index]]

                spikes = load_spikes_data(audio_label, signal, freqs)

                classes = network.input_full_data_spikes(spikes)
                if signal == compare_epochs_by:
                    np.savez_compressed(f'output_spikes/{signal}_{audio_label}_{epoch}.npz',
                                        post_spikes=neuron.out_spikes(),
                                        membrane_potential=neuron.membrane_potential_graph()
                                        )
                network.reset_learning()
                network.reset_input()
                weight_generations_buffer[i + 1, :] = neuron.synapses_weights
                weight_generations = weight_generations_buffer[:i + 2, :]

                np.savez_compressed(f'neurons_weights/{audio_label}_synapses_weights_generations.npz',
                                    synapses_weights=weight_generations)
                l1 = np.sum(np.abs(weight_generations_buffer[i + 1, :] - weight_generations_buffer[i, :]))
                pbar.set_description(
                    f"l1 {l1:.2f} |{printable_weights(neuron.synapses_weights)}| spikes {np.sum(classes)}")
                pbar.update()

            # check if didn't learn much from this epoch
            l1 = np.sum(
                np.abs(weight_generations_buffer[i + 1, :] - weight_generations_buffer[i + 1 - len(signals), :]))
            if epoch > 0 and l1 < .05:
                return weight_generations_buffer
            print(f'\n{neuron.synapses_weights}')
            neuron.set_stdp_ltp(neuron.stdp.A_LTP * 0.95)
            neuron.set_stdp_ltd(neuron.stdp.A_LTD * 0.95)
    return weight_generations_buffer


def learn_neurons(freqs, label, clk_freq):
    network = create_learning_network(clk_freq, freqs)
    neuron = network.neurons[-1]

    seed = sum(map(ord, label))
    train_signals, _ = train_test_files(label, train_test_split=.8, seed=seed)
    prev_weights = np.copy(neuron.synapses_weights)
    print(f'pre - learning - {label}\r\n{prev_weights}')
    make_neuron_learn(
        network,
        label,
        signals=train_signals,
        epochs=20,
    )
    print(f'post - learning - {label}\r\n{neuron.synapses_weights.tolist()}')
    print(
        f'diff - {label}\r\n{dict(zip(freqs, (neuron.synapses_weights - prev_weights).astype(np.int16)))}')

    print('Done.')


def test_neurons(freqs, label, clk_freq):
    print(f'Classify test on {label}')
    network = create_test_network(clk_freq)
    # plot_network(network)

    network.log_out_spikes(-1)
    network.log_out_spikes(-2)
    network.log_out_spikes(-3)

    success = 0

    seed = sum(map(ord, label))
    _, test_signals = train_test_files(label, train_test_split=.8, seed=seed)

    for file_name in test_signals:
        spikes = load_spikes_data(label, file_name, freqs)

        is_success = test_input(network, spikes) == label
        success += is_success

        network.reset_input()

    print(f'success rate on {label} is {success / len(test_signals) * 100}%')


def train_test_files(label, train_test_split=.5, seed=42):
    files_names = np.array(os.listdir(f"../datasets/RWCP_spikes/{label}"))

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
        751, 1046, 1235, 3934, 5478
    ]

    semi_supervised_learning(freqs, clk_freq, freqs * 3)
    # learn_neurons(freqs, 'bottle1', clk_freq)
    # learn_neurons(freqs, 'buzzer', clk_freq)
    # learn_neurons(freqs, 'phone4', clk_freq)

    # test_neurons(freqs, 'bottle1', clk_freq)
    # test_neurons(freqs, 'buzzer', clk_freq)
    # test_neurons(freqs, 'phone4', clk_freq)
