import os

import librosa
import numpy as np
from tqdm import tqdm

from helpers import timing, njit, numbaList
from snn import resonator as sctn


@timing
@njit
def input_to_resonators(resonators, data):
    for potential in data:
        for resonator in resonators:
            sctn.input_by_potential(resonator, potential)


if __name__ == '__main__':
    for audio_type in os.listdir('../sounds/RWCP_resampled'):
        print(f'transform {audio_type}')
        for audio_file in tqdm(os.listdir(f'../sounds/RWCP_resampled/{audio_type}')):
            clk_freq = 1.536 * (10 ** 6)

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
            resonators = numbaList([sctn.CustomResonator(freq0, clk_freq, LF, LP) for (freq0, LF, LP) in freqs])

            audio_path = f"../sounds/RWCP_resampled/{audio_type}/{audio_file}"
            data, sr = librosa.load(audio_path, sr=16000)

            # resample to 1.53M
            data = librosa.resample(data, orig_sr=sr, target_sr=clk_freq, res_type='linear')
            data = data / np.max(data)
            [resonator.network.log_out_spikes(4) for resonator in resonators]
            input_to_resonators(resonators, data)
            spikes = []
            for resonator in resonators:
                neuron = resonator.network.neurons[4]
                spikes.append(neuron.out_spikes[:neuron.index])
            spikes = np.array(spikes)
            output_file = f"../sounds/RWCP_spikes/{audio_type}/{audio_file}"[:-4] + '.npy'
            with open(output_file, 'wb') as f:
                np.save(f, spikes)


