import numpy as np

from tqdm import tqdm
from itertools import product
from helpers.graphs import plot_network
from helpers import timing, load_audio_data
from scripts.rwcp_resonators import snn_based_resonator_for_learning, snn_based_resonator_for_test


def test_input(network, data):
    labels = np.array(['bells5', 'bottle1', 'buzzer', 'phone4'])
    classes = network.input_full_data(data)
    choose_labeled = np.argmax(classes)
    print(f'Find class {labels[choose_labeled]}. out spikes {labels} = {classes}')
    return labels[choose_labeled]


@timing
def make_neuron_learn(network, audio_label, samples, repeats):
    neuron = network.neurons[-1]
    weight_generations = np.zeros((repeats * samples + 1, *neuron.synapses_weights.shape))
    weight_generations[0, :] = neuron.synapses_weights
    for i, (_, sample) in tqdm(list(enumerate(product(range(repeats), range(samples))))):
        try:
            audio_file = f'{(sample % 1000) // 100}{(sample % 100) // 10}{sample % 10}'
            audio_path = f"../sounds/RWCP/{audio_label}/{audio_file}.raw"
            data = load_audio_data(audio_path, clk_freq)
        except FileNotFoundError:
            continue

        network.input_full_data(data)
        network.reset_learning()
        weight_generations[i + 1, :] = neuron.synapses_weights

        np.savez_compressed(f'neurons_weights/{audio_label}_synapses_weights_generations.npz',
                            synapses_weights=weight_generations)
        mse = np.sum(weight_generations[i + 1, :] - weight_generations[i, :]) ** 2
        if mse < 1:
            return weight_generations
    return weight_generations


def learn_neurons(freqs, clk_freq):
    network = snn_based_resonator_for_learning(freqs, clk_freq)
    plot_network(network)

    learning_neuron = network.neurons[-1]

    # for label in ['bells5', 'bottle1', 'buzzer', 'phone4']:
    for label in ['bells5']:
        prev_weights = np.copy(learning_neuron.synapses_weights)
        print(f'pre - learning - {label}\r\n{prev_weights}')
        make_neuron_learn(network, label, samples=20, repeats=10)
        print(f'post - learning - {label}\r\n{learning_neuron.synapses_weights.tolist()}')
        print(
            f'diff - {label}\r\n{dict(zip(freqs, (learning_neuron.synapses_weights - prev_weights).astype(np.int16)))}')


def test_neurons(freqs, audio, clk_freq):
    print(f'Classify test on {audio}')
    network = snn_based_resonator_for_test(freqs, clk_freq)
    # plot_network(network)
    success = 0
    count_test = 0
    for sample in range(25, 30):
        try:
            audio_file = f'{(sample % 1000) // 100}{(sample % 100) // 10}{sample % 10}'
            audio_path = f"../sounds/RWCP/{audio}/{audio_file}.raw"
            data = load_audio_data(audio_path, clk_freq)
        except FileNotFoundError:
            continue

        is_success = test_input(network, data) == audio
        success += is_success
        count_test += 1

    print(f'success rate on {audio} is {success / count_test}')


if __name__ == '__main__':
    clk_freq = int(1.536 * (10 ** 6) * 2)
    freqs = [
        200, 236, 278, 328, 387, 457,
        637, 751, 887, 1046, 1235, 1457,
        1719, 2029, 2825, 3934, 5478 # 3334
    ]
    # learn_neurons(freqs, clk_freq)
    # test_neurons(freqs, 'bottle1', clk_freq)
    # test_neurons(freqs, 'bells5', clk_freq)
    # test_neurons(freqs, 'buzzer', clk_freq)
    test_neurons(freqs, 'phone4', clk_freq)
