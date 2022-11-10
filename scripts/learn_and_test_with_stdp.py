import numpy as np

from tqdm import tqdm
from itertools import product
from helpers.graphs import plot_network
from helpers import timing, load_audio_data
from scripts.rwcp_resonators import snn_based_resonator_for_learning, snn_based_resonator_for_test
from snn.spiking_network import get_labels


def test_input(network, data):
    labels = get_labels(network)
    classes = network.input_full_data(data)
    choose_labeled = np.argmax(classes)
    print(f'Find class {labels[choose_labeled]}. out spikes {labels} = {classes}')
    return labels[choose_labeled]


@timing
def make_neuron_learn(network, audio_label, samples, repeats, continue_learning):
    neuron = network.neurons[-1]
    weight_generations_buffer = np.zeros((repeats * samples + 1, *neuron.synapses_weights.shape))
    weight_generations_buffer[0, :] = neuron.synapses_weights
    i_repeats_samples = list(enumerate(product(range(repeats), range(samples))))
    if continue_learning:
        previous_learning = np.load(f'neurons_weights/{audio_label}_synapses_weights_generations.npz')
        previous_learning = previous_learning['synapses_weights']
        previous_learning = previous_learning[~np.all(previous_learning == 0, axis=1)]
        weight_generations_buffer[:len(previous_learning), :] = previous_learning
        i_repeats_samples = i_repeats_samples[len(previous_learning):]

    is_silence = audio_label == 'silence'

    for i, (_, sample) in tqdm(i_repeats_samples):
        try:
            audio_file = f'{(sample % 1000) // 100}{(sample % 100) // 10}{sample % 10}'
            audio_path = f"../sounds/RWCP/" \
                         f"{'bottle1' if is_silence else audio_label}/" \
                         f"{audio_file}.raw"
            data = load_audio_data(audio_path, clk_freq)
        except FileNotFoundError:
            continue

        data = data[(data > 1e-3) ^ is_silence]
        network.input_full_data(data)
        network.reset_learning()
        weight_generations_buffer[i + 1, :] = neuron.synapses_weights
        weight_generations = weight_generations_buffer[:i + 2, :]

        np.savez_compressed(f'neurons_weights/{audio_label}_synapses_weights_generations.npz',
                            synapses_weights=weight_generations)
        mse = np.sum(weight_generations_buffer[i + 1, :] - weight_generations_buffer[i, :]) ** 2
        if mse < 1:
            return weight_generations_buffer
    return weight_generations_buffer


def learn_neurons(freqs, label, clk_freq, continue_learning=False):
    network = snn_based_resonator_for_learning(freqs, clk_freq)
    plot_network(network)

    learning_neuron = network.neurons[-1]
    network.log_out_spikes(-1)
    for n in network.layers_neurons[-2].neurons:
        network.log_out_spikes(n._id)

    prev_weights = np.copy(learning_neuron.synapses_weights)
    print(f'pre - learning - {label}\r\n{prev_weights}')
    make_neuron_learn(network, label, samples=20, repeats=4, continue_learning=continue_learning)
    print(f'post - learning - {label}\r\n{learning_neuron.synapses_weights.tolist()}')
    print(
        f'diff - {label}\r\n{dict(zip(freqs, (learning_neuron.synapses_weights - prev_weights).astype(np.int16)))}')

    spikes_length = learning_neuron.index
    n_of_neurons = len(network.layers_neurons[-2].neurons) + 1
    spikes_output = np.zeros((n_of_neurons, spikes_length), dtype=np.int8)
    for i, neuron in enumerate(network.layers_neurons[-2].neurons):
        spikes_output[i, :] = neuron.out_spikes[:spikes_length]

    spikes_output[-1, :] = learning_neuron.out_spikes[:spikes_length]

    np.savez_compressed(f'spikes_output_bottle.npz',
                        synapses_weights=spikes_output)
    print('Done.')


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

        data = data[(data > 1e-3)]
        is_success = test_input(network, data) == audio
        success += is_success
        count_test += 1

    print(f'success rate on {audio} is {success / count_test}')


if __name__ == '__main__':
    clk_freq = int(1.536 * (10 ** 6) * 2)
    freqs = [
        200, 236, 278, 328, 387, 457,
        637, 751, 887, 1046, 1235, 1457,
        1719, 2029, 2825, 3334, 3934, 5478
    ]

    # learn_neurons(freqs, 'bells5', clk_freq)
    # learn_neurons(freqs, 'bottle1', clk_freq)
    # learn_neurons(freqs, 'buzzer', clk_freq)
    # learn_neurons(freqs, 'phone4', clk_freq)
    # learn_neurons(freqs, 'silence', clk_freq)

    # test_neurons(freqs, 'bottle1', clk_freq)
    # test_neurons(freqs, 'bells5', clk_freq)
    # test_neurons(freqs, 'buzzer', clk_freq)
    test_neurons(freqs, 'phone4', clk_freq)
