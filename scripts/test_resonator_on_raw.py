
import numpy as np
from matplotlib import pyplot as plt

from snn.graphs import plot_network
from utils import load_audio_data
from scripts.rwcp_resonators import snn_based_resonators


if __name__ == '__main__':
    label = 'buzzer'
    for audio_file in ['00']:
        audio_file = f'0{audio_file}'
        clk_freq = int(1.536 * (10 ** 6)) * 2

        frequencies = [int(200 * (1.18 ** i)) for i in range(0, 19)] + [5478]

        audio_path = f"../sounds/RWCP/{label}/{audio_file}.raw"
        data = load_audio_data(audio_path, clk_freq)
        data = data[:len(data)//2]
        network = snn_based_resonators(frequencies, clk_freq)
        last_layer_neurons = network.layers_neurons[-1].neurons
        plot_network(network)
        for neuron in last_layer_neurons:
            network.log_out_spikes(neuron._id)

        network.input_full_data(data)

        cols = int(np.floor(np.sqrt(len(last_layer_neurons))))
        rows = int(np.ceil(np.sqrt(len(last_layer_neurons))))

        fig, axs = plt.subplots(cols, rows, sharex='all', sharey='all')
        fig.tight_layout(pad=.8)
        for i, neuron in enumerate(last_layer_neurons):
            spikes_amount = np.convolve(neuron.out_spikes[:neuron.index], np.ones(1000, dtype=int), 'valid')
            axs[i//rows, i % rows].plot(spikes_amount)
            axs[i//rows, i % rows].set_title(f'{neuron.label}')
            axs[i//rows, i % rows].set_yticks([0, 15, 30, 45])

        plt.suptitle(f'spikes for {label} in window of 1000')
        plt.figure(figsize=(24, 18), dpi=80)
        # plt.savefig(f'../plots/{audio_file}.png')
        plt.show()
