import multiprocessing
import os
from pathlib import Path

import pandas as pd
import numpy as np

from utils import load_audio_data
from scripts.rwcp_resonators import snn_based_resonators


def generate_spikes(audio_label: str, audio_file: str):
    clk_freq = int(1.536 * (10 ** 6) * 2)
    freqs = [
        200, 236, 278, 328, 387, 457,
        751, 887, 1046, 1235, 1457,
        1719, 2029, 2825, 3934, 5478
    ]

    try:
        audio_path = f"../datasets/RWCP/" \
                     f"{audio_label}/" \
                     f"{audio_file}"
        data = load_audio_data(audio_path, clk_freq, remove_silence=True)
    except FileNotFoundError:
        return

    network = snn_based_resonators(freqs, clk_freq)
    output_neurons = network.layers_neurons[-1].neurons
    for n in output_neurons:
        network.log_out_spikes(n._id)
    network.input_full_data(data)

    output_spikes = {
        n.label: n.out_spikes()
        for n in output_neurons
    }
    Path(f'../datasets/RWCP_spikes/{audio_label}').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        file=f'../datasets/RWCP_spikes/{audio_label}/{audio_file}.npz',
        **output_spikes
    )


if __name__ == '__main__':
    # filter_labels = ['bottle1', 'buzzer', 'phone4']
    filter_labels = [
        'dice1', 'metal05',
        'cherry1', 'bottle2'
    ]

    args = [
        (audio_label, audio_file)
        for audio_label in filter_labels
        for audio_file in os.listdir(f'../datasets/RWCP/{audio_label}')
    ]

    # for label in filter_labels:
    #     print(f'Start {label}')
    #     generate_spikes(label, '000.raw')
    with multiprocessing.Pool(processes=6) as pool:
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
