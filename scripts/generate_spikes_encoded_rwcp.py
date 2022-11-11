import multiprocessing
import os
from pathlib import Path

import pandas as pd
import tqdm
import numpy as np

from datasets.RWCP_spikes import RWCPSpikesDataset
from helpers import load_audio_data
from scripts.rwcp_resonators import snn_based_resonators


def generate_spikes(audio_label: str, audio_file: str):
    clk_freq = int(1.536 * (10 ** 6) * 2)
    freqs = [
        200, 236, 278, 328, 387, 457,
        637, 751, 887, 1046, 1235, 1457,
        1719, 2029, 2825, 3334, 3934, 5478
    ]

    try:
        audio_path = f"../datasets/RWCP/" \
                     f"{audio_label}/" \
                     f"{audio_file}"
        data = load_audio_data(audio_path, clk_freq)
    except FileNotFoundError:
        return

    data = data[data > 1e-3]
    network = snn_based_resonators(freqs, clk_freq)
    output_neurons = network.layers_neurons[-1].neurons
    for n in output_neurons:
        network.log_out_spikes(n._id)
    network.input_full_data(data)

    output_spikes = {
        n.label: n.out_spikes[:n.index]
        for n in output_neurons
    }
    Path(f'../datasets/RWCP_spikes/{audio_label}').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        file=f'../datasets/RWCP_spikes/{audio_label}/{audio_file}.npz',
        **output_spikes
    )


if __name__ == '__main__':
    filter_labels = ['bells5', 'bottle1', 'buzzer', 'phone4']

    args = [(audio_label, audio_file)
            for audio_label in os.listdir('../datasets/RWCP')
            for audio_file in os.listdir(f'../datasets/RWCP/{audio_label}')
            if audio_label in filter_labels]

    with multiprocessing.Pool(processes=12) as pool:
        pool.starmap(generate_spikes, args)

    meta_data = [
        {
            'label': audio_label,
            'file_name': audio_file,
            'size': len(np.load(f'../datasets/RWCP_spikes/'
                                f'{audio_label}/'
                                f'{audio_file}')['f200']),

        }
        for audio_label in os.listdir('../datasets/RWCP_spikes')
        if os.path.isdir(f'../datasets/RWCP_spikes/{audio_label}')
        for audio_file in os.listdir(f'../datasets/RWCP_spikes/{audio_label}')
    ]
    pd.DataFrame(meta_data).to_csv('../datasets/RWCP_spikes/meta_data.csv',
                                   index=False)
