import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List, Tuple
from snn.layers import SCTNLayer
from snn.spiking_network import SpikingNetwork
from utils import neurons_labels, save_network_weights, load_network_weights
from scripts.rwcp_resonators import create_neuron_for_labeling


def create_random_network(freqs, clk_freq, n_neurons):
    # create network with n neurons as labels.
    # The neurons are not learning yet.
    network = SpikingNetwork(clk_freq)
    labels_neurons = [
        create_neuron_for_labeling(np.random.random(len(freqs)) * 10 + 30)
        for _ in range(n_neurons)
    ]
    network.add_layer(SCTNLayer(labels_neurons))
    for neuron in network.layers_neurons[-1].neurons:
        network.log_out_spikes(neuron._id)
    return network


def train_test_files(label, train_ratio=.5, seed=42):
    files_names = np.array(os.listdir(f"../datasets/RWCP_spikes/{label}"))

    np.random.seed(seed)
    shuffle = np.random.permutation(len(files_names))
    files_names = files_names[shuffle]
    train = files_names[:int(len(files_names) * train_ratio)]
    test = files_names[int(len(files_names) * train_ratio):]
    return train, test


def get_signals(test: bool, seed=42, train_ratio=.5, oversample=False) -> List[Tuple[str, str]]:
    bottle_files, bottle_test_files = train_test_files('bottle1', seed=seed, train_ratio=train_ratio)
    buzzer_files, buzzer_test_files = train_test_files('buzzer', seed=seed, train_ratio=train_ratio)
    phone_files, phone_test_files = train_test_files('phone4', seed=seed, train_ratio=train_ratio)

    np.random.seed(seed)

    if test:
        bottle_files = bottle_test_files
        buzzer_files = buzzer_test_files
        phone_files = phone_test_files

    if oversample:
        labels_files = [bottle_files, buzzer_files, phone_files]
        max_samples = max(map(len, labels_files))

        def oversample(files):
            extra_samples = max_samples - len(files)
            choices = np.random.choice(len(files), extra_samples)
            return np.concatenate([files, files[choices]])

        bottle_files = oversample(bottle_files)
        buzzer_files = oversample(buzzer_files)
        phone_files = oversample(phone_files)

    signals_files = [f'bottle1/{f}' for f in bottle_files] + \
                    [f'buzzer/{f}' for f in buzzer_files] + \
                    [f'phone4/{f}' for f in phone_files]
    labels = ['bottle1'] * len(bottle_files) + \
             ['buzzer'] * len(buzzer_files) + \
             ['phone4'] * len(phone_files)

    res = np.array(list(zip(signals_files, labels)))

    shuffle = np.random.permutation(len(res))
    return res[shuffle]


def activate_stdp_to_same_label_neurons(network: SpikingNetwork,
                                        label: str):
    time_to_learn = 2.5e-3
    tau = network.clk_freq * time_to_learn / 2

    for neuron in network.layers_neurons[-1].neurons:
        if neuron.label == label:
            neuron.set_stdp(0.00008, -0.00008, tau, clk_freq, 70, -20)


def activate_stdp_to_different_label_neurons(network: SpikingNetwork,
                                             label: str):
    time_to_learn = 2.5e-3
    tau = network.clk_freq * time_to_learn / 2

    for neuron in network.layers_neurons[-1].neurons:
        if neuron.label != label and neuron.label is not None:
            neuron.set_stdp(-0.00004, 0, tau, clk_freq, 70, -20)


def load_spikes_data(file_name, freqs):
    spikes = pd.DataFrame \
        .from_dict(dict(
        np.load(f'..\datasets\RWCP_spikes\\{file_name}')
    ))
    columns = [f'f{f}' for f in freqs]
    return spikes[columns].to_numpy()


def get_unlabeled_neurons(network: SpikingNetwork) -> np.ndarray:
    unlabeled_neurons = [neuron.label is None
                         for neuron in network.layers_neurons[-1].neurons]
    return np.array(unlabeled_neurons, dtype=np.int8)


def tag_neuron_a_label(network, post_spikes, label):
    while True:
        arg_most_active_neuron = np.argmax(post_spikes)
        most_active_neuron = network.layers_neurons[-1].neurons[arg_most_active_neuron]
        if most_active_neuron.label is None:
            most_active_neuron.label = label
            return True
        elif most_active_neuron.label == label:
            return False
        post_spikes[arg_most_active_neuron] = -1


def learning_process(network: SpikingNetwork,
                     train_signals: List[Tuple[str, str]],
                     epochs: int,
                     l1_stop: float = .1):
    neurons_encoder = {
        None: '-',
        'bottle1': '🍶',
        'buzzer': '🚨',
        'phone4': '📱'
    }
    total_runs = epochs * len(train_signals)

    labels_neurons = network.layers_neurons[-1].neurons
    weight_generations_buffer = np.zeros((epochs * len(train_signals) + 1,
                                          len(labels_neurons),
                                          len(labels_neurons[0].synapses_weights)))

    for n, neuron in enumerate(labels_neurons):
        weight_generations_buffer[0, n, :] = neuron.synapses_weights

    count_labels = {
        'bottle1': len(labels_neurons) // 3,
        'buzzer': len(labels_neurons) // 3,
        'phone4': len(labels_neurons) // 3
    }
    with tqdm(total=total_runs) as pbar:
        for epoch in range(epochs):
            permutation_audio_file_indices = np.random.permutation(len(train_signals))
            for signal_index in range(len(train_signals)):
                i = epoch * len(train_signals) + signal_index
                signal, label = train_signals[permutation_audio_file_indices[signal_index]]
                activate_stdp_to_same_label_neurons(network, label)
                activate_stdp_to_different_label_neurons(network, label)

                spikes = load_spikes_data(signal, freqs)

                post_spikes = network.input_full_data_spikes(spikes)
                # post_spikes *= get_unlabeled_neurons(network)
                if count_labels[label] > 0 and tag_neuron_a_label(network, post_spikes, label):
                    # if new neuron got a label, count it.
                    count_labels[label] -= 1

                for n, neuron in enumerate(labels_neurons):
                    weight_generations_buffer[i + 1, n, :] = neuron.synapses_weights

                # prepare for new input
                network.reset_learning()
                network.reset_input()

                l1 = np.sum(
                    np.abs(
                        weight_generations_buffer[i + 1, :, :] -
                        weight_generations_buffer[epoch * len(train_signals), :, :]
                    )
                )
                pbar.set_description(
                    f"l1 {l1:.2f} |{neurons_labels(network.layers_neurons[-1].neurons, encoder=neurons_encoder)}")
                pbar.update()

            save_network_weights(network, path='neurons_weights/semi_supervised_learning.pickle')
            l1 = np.sum(
                np.abs(
                    weight_generations_buffer[(epoch + 1) * len(train_signals), :] -
                    weight_generations_buffer[epoch * len(train_signals), :]
                )
            )
            if epoch > 0 and l1 < l1_stop:
                return


def test_process(network, test_signal_files):
    labels = [f'{n.label}_{n._id}' for n in network.layers_neurons[-1].neurons]
    predict_results = []
    for signal_file, label in tqdm(test_signal_files):
        spikes = load_spikes_data(signal_file, freqs)
        post_spikes = network.input_full_data_spikes(spikes)
        res = dict(zip(labels, post_spikes))
        res['label'] = label
        predict_results.append(res)
        network.reset_input()

    df = pd.DataFrame.from_records(predict_results)
    df.to_csv('output_spikes/semi_supervised_test.csv', index=False)
    return df


def semi_supervised_learning(freqs, clk_freq, n_neurons):
    network = create_random_network(freqs, clk_freq, n_neurons)
    train_signals_files = get_signals(test=False, train_ratio=.5, oversample=True)
    learning_process(network, train_signals_files, epochs=10, l1_stop=.1)
    save_network_weights(network, path='neurons_weights/semi_supervised_learning.pickle')


def test_network(network=None,
                 freqs=None,
                 clk_freq=None,
                 n_neurons=None,
                 train_ratio=.5):
    if network is None:
        network = create_random_network(freqs, clk_freq, n_neurons)
        load_network_weights(network, path='neurons_weights/semi_supervised_learning.pickle')
    test_signals_files = get_signals(test=True, train_ratio=train_ratio, oversample=False)
    return test_process(network, test_signals_files)


if __name__ == '__main__':
    clk_freq = int(1.536 * (10 ** 6) * 2)

    freqs = [
        751, 1046, 1235, 3934, 5478
    ]

    # semi_supervised_learning(freqs, clk_freq, len(freqs) * 6)
    test_network(
        freqs=freqs,
        clk_freq=clk_freq,
        n_neurons=len(freqs) * 6
    )
