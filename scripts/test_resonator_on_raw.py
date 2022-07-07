import wave

import librosa
import numpy as np
from matplotlib import pyplot as plt

from helpers import timing, njit, numbaList
from helpers.graphs import plot_network
from networks.sdsp_resonators import snn_based_resonator
from snn import resonator as sctn


@timing
@njit
def input_to_resonators(resonators, data):
    for potential in data:
        for resonator in resonators:
            sctn.input_by_potential(resonator, potential)


if __name__ == '__main__':
    for audio_file in [10]:
        audio_file = f'0{audio_file}'
        clk_freq = 1.536 * (10 ** 6)
        # freqs = np.arange(20, 10_000, 10_000//15)
        freqs = [
            (100, 3, 299),
            (250, 4, 60),
            (500, 5, 14),
            (1000, 6, 3),
            (1750, 6, 1),
            (2800, 3, 10),
            (3500, 3, 8),
            (5000, 3, 4),
            (7500, 4, 1),
            (10000, 3, 2),
            (15000, 3, 1),
            ]
        resonators = numbaList([sctn.CustomResonator(freq[0], clk_freq, freq[1], freq[2]) for freq in freqs])
        [resonator.network.log_membrane_potential(4) for resonator in resonators]

        audio_path = f"../sounds/RWCP/buzzer/{audio_file}.raw"
        with open(audio_path, "rb") as inp_f:
            data = inp_f.read()
            with wave.open("sound.wav", "wb") as out_f:
                out_f.setnchannels(1)
                out_f.setsampwidth(2)  # number of bytes
                out_f.setframerate(16000)
                out_f.writeframesraw(data)
        audio_path = 'sound.wav'

        data, sr = librosa.load(audio_path, sr=16000)

        # resample to 1.53M
        data = librosa.resample(data, orig_sr=sr, target_sr=clk_freq, res_type='linear')
        data = data / np.max(data)

        network = snn_based_resonator(freqs)
        plot_network(network)

        with open('sound_resampled.raw', 'wb') as f:
            f.write(data.tobytes())
        input_to_resonators(resonators, data)

        neuron = network.neurons[-1]
        membrane = neuron.membrane_potential_graph[:neuron.index]
        plt.plot(membrane)
        plt.show()
        # for n in [4]:#range(1, 18, 2):
        #     for resonator in resonators:
        #         neuron = resonator.network.neurons[n]
        #         # spikes_amount = np.convolve(neuron.out_spikes[:neuron.index], np.ones(1000, dtype=int), 'valid')
        #         # plt.plot(spikes_amount, label=resonator.freq0)
        #         plt.title(f'audio file {audio_file} freq {resonator.freq0}')
        #         plt.plot(neuron.membrane_potential_graph[:neuron.index], label=resonator.freq0)
        #         plt.show()
        # plt.legend()
        # plt.title(f'audio_file {audio_file}')
        # plt.show()
        # plt.savefig(f'../plots/.png')
