import json
import wave

import librosa
import numpy as np
from matplotlib import pyplot as plt

from helpers import timing, njit
from helpers.graphs import plot_network
from scripts.rwcp_resonators import snn_based_resonator
from snn import resonator as sctn
from snn.resonator import create_excitatory_resonator, create_excitatory_inhibitory_resonator


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

        frequencies = [int(200 * (1.18 ** i)) for i in range(0, 19)] + [5478]
        resonators = []
        for freq0 in frequencies:
            resonator = create_excitatory_inhibitory_resonator(freq0=freq0, clk_freq=clk_freq)
            resonator.network.log_out_spikes(-1)
            resonators.append(resonator)

        label = 'bells5'
        audio_path = f"../sounds/RWCP/{label}/{audio_file}.raw"
        with open(audio_path, "rb") as inp_f:
            data = inp_f.read()
            with wave.open("sound.wav", "wb") as out_f:
                out_f.setnchannels(1)
                out_f.setsampwidth(2)  # number of bytes
                out_f.setframerate(16000)
                out_f.writeframesraw(data)
        audio_path = 'sound.wav'

        data, sr = librosa.load(audio_path, sr=16000)

        # resample to clk_freq
        data = librosa.resample(data, orig_sr=sr, target_sr=clk_freq, res_type='linear')
        # data = data[np.abs(data) > 2e-3]
        data = data / np.max(data)

        network = snn_based_resonator(frequencies, clk_freq)
        plot_network(network)

        with open('sound_resampled.raw', 'wb') as f:
            f.write(data.tobytes())
        # data = np.concatenate([np.zeros(len(data)//2), data])
        input_to_resonators(resonators, data)
        cols = int(np.floor(np.sqrt(len(resonators))))
        rows = int(np.ceil(np.sqrt(len(resonators))))

        fig, axs = plt.subplots(cols, rows, sharex='all', sharey='all',)
        fig.tight_layout(pad=.8)
        for i, resonator in enumerate(resonators):
            neuron = resonator.network.neurons[-1]
            spikes_amount = np.convolve(neuron.out_spikes[:neuron.index], np.ones(1000, dtype=int), 'valid')
            axs[i//rows, i % rows].plot(spikes_amount)
            axs[i//rows, i % rows].set_title(f'{resonator.freq0}')
            axs[i//rows, i % rows].set_yticks([0, 50, 100])
            # plt.plot(spikes_amount, label=resonator.freq0, alpha=0.65)
            # plt.title(f'audio file {audio_file} freq {resonator.freq0}')
            # membrane = neuron.membrane_potential_graph()
            # plt.text(x=len(membrane)//2,
            #          y=membrane[len(membrane)//2],
            #          s=resonator.freq0)
            # plt.plot(membrane, label=resonator.freq0)
            # plt.show()

        plt.suptitle(f'spikes for {label} in window of 1000')
        plt.figure(figsize=(24, 18), dpi=80)
        # plt.savefig(f'../plots/{audio_file}.png')
        plt.show()
