import os
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

from snn.layers import SCTNLayer
from helpers.graphs import plot_network
from snn.spiking_network import SpikingNetwork
from scripts.rwcp_resonators import learning_neuron


def make_neuron_learn(network, audio_label, samples, epoch):
    learning_layer_neurons = network.layers_neurons[-1].neurons
    weight_generations_buffer = {
        neuron._id: np.zeros((epoch * samples + 1, *neuron.synapses_weights.shape))
        for neuron in learning_layer_neurons
    }
    for neuron in learning_layer_neurons:
        weight_generations_buffer[neuron._id][0, :] = neuron.synapses_weights

    i_repeats_samples = list(
        enumerate(
            product(
                range(epoch),
                os.listdir(f'../datasets/RWCP_spikes/{audio_label}')
            )
        )
    )
    finish_learn = {
        neuron._id: False
        for neuron in learning_layer_neurons
    }
    with tqdm(total=2 * len(learning_layer_neurons) * len(i_repeats_samples)) as pbar:
        for i, (_, f_name) in tqdm(i_repeats_samples):
            data = np.load(f'../datasets/RWCP_spikes/{audio_label}/{f_name}')

            # to numpy shaped as (timestamp, resonators)
            data = np.array(list(dict(data).values())).T
            for batch in range(len(learning_layer_neurons) * 2):
                data_batch = data[(batch // 2) * len(data):
                                  (batch // 2 + 1) * len(data)]
                network.input_full_data_spikes(data_batch)
                network.reset_learning()

                for neuron in learning_layer_neurons:
                    if finish_learn[neuron._id]:
                        continue
                    weight_generations_buffer[neuron._id][i + 1, :] = neuron.synapses_weights
                    weight_generations = weight_generations_buffer[:i + 2, :]
                    np.savez_compressed(
                        f'neurons_weights/{audio_label}_neuron{neuron._id}_synapses_weights_generations.npz',
                        synapses_weights=weight_generations)
                    mse = np.sum(weight_generations_buffer[neuron._id][i + 1, :]
                                 - weight_generations_buffer[neuron._id][i, :]) ** 2
                    finish_learn[neuron._id] = mse < 1

                pbar.update(1)
    return weight_generations_buffer


def learn_neurons(freqs, label, clk_freq):
    network = SpikingNetwork(clk_freq)
    df = pd.read_csv('../datasets/RWCP_spikes/meta_data.csv')
    mean_time_s = np.mean(df.loc[df['label'] == label, 'size']) / clk_freq
    n_of_neurons = int(mean_time_s * 10)
    list_of_neurons = [learning_neuron(freqs, clk_freq) for _ in range(n_of_neurons)]
    learning_layer = SCTNLayer(list_of_neurons)
    network.add_layer(learning_layer, True, True)
    plot_network(network)

    make_neuron_learn(network, label, samples=20, epoch=4)
    print('Done.')


if __name__ == '__main__':
    clk_freq = int(1.536 * (10 ** 6) * 2)
    freqs = [
        200, 236, 278, 328, 387, 457,
        751, 887, 1046, 1235, 1457,
        1719, 2029, 2825, 3934, 5478
    ]

    learn_neurons(freqs, 'bells5', clk_freq)
    # learn_neurons(freqs, 'bottle1', clk_freq)
    # learn_neurons(freqs, 'buzzer', clk_freq)
    # learn_neurons(freqs, 'phone4', clk_freq)
    # learn_neurons(freqs, 'silence', clk_freq)
