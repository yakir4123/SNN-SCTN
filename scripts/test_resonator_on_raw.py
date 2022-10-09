import json
import wave

import librosa
import numpy as np
from matplotlib import pyplot as plt

from helpers import timing, njit, numbaList
from helpers.graphs import plot_network
from networks.sdsp_resonators import snn_based_resonator
from snn import resonator as sctn
from snn.resonator import OptimizationResonator


@timing
@njit
def input_to_resonators(resonators, data):
    for potential in data:
        for resonator in resonators:
            sctn.input_by_potential(resonator, potential)


if __name__ == '__main__':
    for audio_file in ['00']:
        audio_file = f'0{audio_file}'
        clk_freq = int(1.536 * (10 ** 6)) * 2

        frequencies = [int(200 * (1.18 ** i)) for i in range(0, 19)]
        resonators = []
        for freq0 in frequencies:
            with open(f'../filters/clk_{clk_freq}/parameters/f_{freq0}.json') as f:
                parameters = json.load(f)
            th_gains = [parameters[f'th_gain{i}'] for i in range(4)]
            weighted_gains = [parameters[f'weight_gain{i}'] for i in range(5)]
            resonator = OptimizationResonator(freq0, clk_freq,
                                              parameters['LF'], parameters['LP'],
                                              th_gains, weighted_gains,
                                              parameters['amplitude_gain'])
            resonator.network.log_membrane_potential(-1)
            resonators.append(resonator)

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

        network = snn_based_resonator(frequencies, clk_freq)
        plot_network(network)

        with open('sound_resampled.raw', 'wb') as f:
            f.write(data.tobytes())
        data = np.concatenate([np.zeros(len(data)//2), data])
        input_to_resonators(resonators, data)

        for resonator in resonators[::-1]:
            neuron = resonator.network.neurons[-1]
            # spikes_amount = np.convolve(neuron.out_spikes[:neuron.index], np.ones(1000, dtype=int), 'valid')
            # plt.plot(spikes_amount, label=resonator.freq0)
            # plt.title(f'audio file {audio_file} freq {resonator.freq0}')
            membrane = neuron.membrane_potential_graph()
            plt.text(x=len(membrane)//2,
                     y=membrane[len(membrane)//2],
                     s=resonator.freq0)
            plt.plot(membrane, label=resonator.freq0)
            # plt.show()
        # plt.legend(ncol=3)
        # plt.figure(figsize=(16, 12), dpi=80)
        plt.savefig(f'../plots/{audio_file}.png')
        # plt.show()
