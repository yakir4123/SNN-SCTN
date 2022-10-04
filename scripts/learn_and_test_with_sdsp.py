import os
import wave

import librosa
import numpy as np
from matplotlib import pyplot as plt

from helpers import timing, njit, numbaList
from helpers.graphs import plot_network
from networks.sdsp_resonators import snn_based_resonator, snn_based_resonator_for_learning, snn_based_resonator_for_test
from snn import resonator as sctn


@timing
@njit
def input_to_resonators(resonators, data):
    for potential in data:
        for resonator in resonators:
            sctn.input_by_potential(resonator, potential)

@njit
def input_to_network(network, data):
    # for i, potential in enumerate(data):
    #     network.input_potential(potential)
    for potential in data:
        network.input_potential(potential)


def test_input(network, data):
    labels = np.array(['bells5', 'bottle1', 'buzzer'])
    classes = np.zeros(3)
    for i, potential in enumerate(data):
        res = network.input_potential(potential)
        classes += res
    choose_labeled = np.argmax(classes)
    print(f'Find class {labels[choose_labeled]}. out spikes {labels} = {classes}')
    return labels[choose_labeled]


def resample_all_raw_files():
    for audio_type in os.listdir('../sounds/RWCP'):
        for audio_file in os.listdir(f'../sounds/RWCP/{audio_type}'):
            audio_path = f"../sounds/RWCP/{audio_type}/{audio_file}"
            target_path = f"../sounds/RWCP_resampled/{audio_type}/{audio_file}"[:-3] + 'wav'
            with open(audio_path, "rb") as inp_f:
                data = inp_f.read()
                with wave.open(target_path, "wb") as out_f:
                    out_f.setnchannels(1)
                    out_f.setsampwidth(2)  # number of bytes
                    out_f.setframerate(16000)
                    out_f.writeframesraw(data)

@timing
def make_neuron_learn(network, audio, samples):
    if audio is None:
        input_to_network(network, np.zeros(76500))  # 50ms

    for i in range(1):
        for sample in range(samples):
            try:
                audio_file = f'{(sample%1000)//100}{(sample%100)//10}{sample%10}'
                audio_path = f"../sounds/RWCP_resampled/{audio}/{audio_file}.wav"

                data, sr = librosa.load(audio_path, sr=16000)
            except FileNotFoundError:
                continue

            clk_freq = 1.536 * (10 ** 6)
            # resample to 1.53M
            data = librosa.resample(data, orig_sr=sr, target_sr=clk_freq, res_type='linear')
            data = data / np.max(data)
            # data = data[:150000]
            input_to_network(network, data)


def learn_neurons(freqs, clk_freq):
    network = snn_based_resonator_for_learning(freqs, clk_freq)
    plot_network(network)

    neuron = network.neurons[-1]

    def clear_neuron(neuron_):
        # make_neuron_learn(network, None, 0)
        neuron_.synapses_weights = np.random.random(len(neuron_.synapses_weights)) * 150 - 75
        neuron_.membrane_potential = 0

    network.log_membrane_potential(-1)
    timing(make_neuron_learn(network, 'bells5', 1))
    plt.plot(network.neurons[-1].membrane_potential_graph())
    print(f'bells5 {neuron.synapses_weights}')
    clear_neuron(neuron)
    timing(make_neuron_learn(network, 'bottle1', 1))
    print(f'bottle1 {neuron.synapses_weights}')
    clear_neuron(neuron)
    timing(make_neuron_learn(network, 'buzzer', 1))
    print(f'buzzer {neuron.synapses_weights}')
    clear_neuron(neuron)


def test_neurons(freqs, audio, clk_freq):
    print(f'Classify test on {audio}')
    network = snn_based_resonator_for_test(freqs)
    plot_network(network)
    # network.log_membrane_potential(66)
    # network.log_membrane_potential(67)
    # network.log_membrane_potential(68)
    success = 0
    count_test = 0
    for sample in range(25, 100):
        try:
            audio_file = f'{(sample%1000)//100}{(sample%100)//10}{sample%10}'
            audio_path = f"../sounds/RWCP_resampled/{audio}/{audio_file}.wav"

            data, sr = librosa.load(audio_path, sr=16000)
        except FileNotFoundError:
            continue

        data = librosa.resample(data, orig_sr=sr, target_sr=clk_freq, res_type='linear')
        data = data / np.max(data)

        is_success = test_input(network, data) == audio
        success += is_success
        count_test += 1
        # if not is_success:
        #     bells_neuron = network.neurons[66]
        #     plt.plot(bells_neuron.membrane_potential_graph[:bells_neuron.index])
        #     plt.title(f'bells neuron on {audio}: {audio_file}')
        #     plt.show()
        #     bottle_neuron = network.neurons[67]
        #     plt.plot(bottle_neuron.membrane_potential_graph[:bottle_neuron.index])
        #     plt.title(f'bottle neuron on {audio}: {audio_file}')
        #     plt.show()
        #     buzzer_neuron = network.neurons[68]
        #     plt.plot(buzzer_neuron.membrane_potential_graph[:buzzer_neuron.index])
        #     plt.title(f'buzzer neuron on {audio}: {audio_file}')
        #     plt.show()

    print(f'success rate on {audio} is {success / count_test}')


if __name__ == '__main__':
    # resample to 1.53M
    clk_freq = 1.536 * (10 ** 6)
    freqs = [int(100 * (1.18 ** i)) for i in range(0, 23)]
    learn_neurons(freqs, clk_freq)
    # test_neurons(freqs, 'bells5', clk_freq)
    # test_neurons(freqs, 'bottle1', clk_freq)
    # test_neurons(freqs, 'buzzer', clk_freq)
