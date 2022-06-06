import wave

import librosa
import numpy as np
from matplotlib import pyplot as plt

from helpers import timing, njit, numbaList
from snn import resonator as sctn


@timing
@njit
def input_to_resonators(resonators, data):
    for potential in data:
        for resonator in resonators:
            sctn.input_by_potential(resonator, potential)


if __name__ == '__main__':
    clk_freq = 1.536 * (10 ** 6)
    freqs = [740, 7000]
    # resonators = numbaList([sctn.SemiResonator(freq0, clk_freq) for freq0 in freqs])
    # resonators = numbaList([sctn.CustomResonator(freq0, clk_freq) for freq0 in freqs])
    resonators = numbaList([sctn.Resonator(freq0, clk_freq) for freq0 in freqs])

    audio_path = "../sounds/RWCP/phone4/000.raw"
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
    with open('sound_resampled.raw', 'wb') as f:
        f.write(data.tobytes())
    input_to_resonators(resonators, data)

    for resonator in resonators:
        for n in range(0, 17):
            neuron = resonator.network.neurons[n]
            spikes_amount = sum(neuron.out_spikes[:neuron.index])
            print(f'{resonator.freq0}: {spikes_amount - len(data) // 2}')
            plt.plot(neuron.membrane_potential_graph[:neuron.index])
            plt.show()
