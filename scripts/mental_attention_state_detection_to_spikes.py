import os
import torch
import pickle
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import resample
from scipy.signal import butter, filtfilt

from tqdm import tqdm
from pathlib import Path
from snn.resonator import trained_resonator


def plot_spectrogram(data, fs, fmin, fmax):
    # plot the spectrogram
    plt.figure(figsize=(14, 5))
    Sxx, freqs, bins, im = plt.specgram(data, NFFT=256, Fs=fs,
                                        noverlap=128, cmap='jet')
    print(bins)
    plt.ylim(fmin, fmax)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    # plt.title(channel + f' Spectrogram {channel_name}({fmin} - {fmax}) Hz')
    plt.imshow(Sxx, aspect='auto',
               cmap='jet', origin='lower',
               extent=[bins[0], bins[-1], freqs[0], freqs[-1]],
               vmin=0, vmax=np.max(Sxx[(freqs >= fmin) & (freqs <= fmax)]))
    plt.colorbar()
    plt.show()


def bp_filter(data, f_lo, f_hi, fs):
    """ Digital band pass filter (6-th order Butterworth)
    Args:
        data: numpy.array, time along axis 0
        (f_lo, f_hi): frequency band to extract [Hz]
        fs: sampling frequency [Hz]
    Returns:
        data_filt: band-pass filtered data, same shape as data """
    data_filt = np.zeros_like(data)
    f_ny = fs / 2.  # Nyquist frequency
    b_lo = f_lo / f_ny  # normalized frequency [0..1]
    b_hi = f_hi / f_ny  # normalized frequency [0..1]
    # band-pass filter parameters
    p_lp = {"N": 6, "Wn": b_hi, "btype": "lowpass", "analog": False, "output": "ba"}
    p_hp = {"N": 6, "Wn": b_lo, "btype": "highpass", "analog": False, "output": "ba"}
    bp_b1, bp_a1 = butter(**p_lp)
    bp_b2, bp_a2 = butter(**p_hp)
    data_filt = filtfilt(bp_b1, bp_a1, data, axis=0)
    data_filt = filtfilt(bp_b2, bp_a2, data_filt, axis=0)
    return data_filt

#  Function to read in the EEG data and extract the valid lead data, low and high pass filter and z-transform the data.
#  Returns a dataframe.
def get_EEG_data(data_root, filename):
    # Extract the data from one of these files.
    hz = 128
    #filename = 'eeg_record30.mat'
    mat = loadmat(f'{data_root}/{filename}')
    data = pd.DataFrame.from_dict(mat["o"]["data"][0,0])

    # Limit the data to the 7 valid EEG leads.
    dat = data.filter(list(range(3, 17)))
    dat.columns = list(range(1, 15))
    dat = dat.filter([1,2, 3, 4,5,6, 7, 8, 9,10,11,12,13,14,17], axis=1)
    labels = ['AF3','F7', 'F3','FC5','T7','P7','O1', 'O2','P8','T8', 'FC6','F4','F8','AF4']  # FP2 should really be AF4
    dat.columns = labels

    # Filter the data, high pass .5 Hz, low pass 62 Hz.
    lo, hi = .5, 62
    # Do the filtering.
    datf = bp_filter(dat.to_numpy(), lo, hi, hz)

    # Convert back to a dataframe.
    dat = pd.DataFrame({c: datf[:, i] for i, c in enumerate(labels)})

    # Z-transform each column
    # dat = dat.apply(zscore)

    return dat

data_root = '../datasets/EEG_data_for_Mental_Attention_State_Detection/EEG_Data/'
files = os.listdir(data_root)


def get_trial_data(trial):
    dat = get_EEG_data(data_root, f'eeg_record{trial}.mat')
    return dat


example_data = get_trial_data(1)
channel_names = example_data.columns

# 5 subjects, each subject did 7 trials. except the 5'th subject that did 6 trials.
#  The first 2 trials were used to get familiar with the process.

subject_map = {i: np.arange(3, 8) + 7 * (i - 1) for i in range(1, 5)}
subject_map[5] = np.arange(3, 7) + 7 * 4


# number of samples in the original and resampled signals
def resample_signal(f_new, f_source, data):
    n_samples_orig = data.shape[0]
    n_samples_new = int(n_samples_orig * f_new / f_source)

    # resample the signal
    return resample(data, n_samples_new)


def generate_spikes(resonator, data_resampled):
    resonator.input_full_data(data_resampled)


def save_output(resonator, spikes_output_path):
    output_neuron = resonator.layers_neurons[-1].neurons[-1]
    np.savez_compressed(
        file=spikes_output_path,
        spikes=output_neuron.out_spikes(False, clk_freq * 60).astype('int8')
    )


def resonator_fname_to_freq(name):
    return float(name[:-4])


def create_datasets(time_of_sample_s, overlap, trials, signal_fs, output_path, skip_beginning_s):
    trials_folder = '../datasets/EEG_data_for_Mental_Attention_State_Detection/EEG_spikes'
    spikes_in_sample = int(signal_fs * time_of_sample_s)
    step = int(spikes_in_sample * (1 - overlap))
    bands = {
        'Delta': (.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 14),
        'Beta': (14, 32),
        'Gamma': (32, 62),
    }
    with tqdm(total=len(trials) * 3 * 7 * (14 * 36) * int(1 / time_of_sample_s)) as pbar:
        for trial in tqdm(trials):
            for label in os.listdir(f'{trials_folder}/{trial}'):
                for minute in os.listdir(f'{trials_folder}/{trial}/{label}'):
                    full_minute_spikes_input = {}
                    minute = int(minute)
                    for band_name, (lf, hf) in bands.items():
                        for ch_name in os.listdir(f'{trials_folder}/{trial}/{label}/{minute}'):
                            full_minute_spikes_input[f'{ch_name}-{band_name}'] = {}
                            for clk_freq_str in os.listdir(f'{trials_folder}/{trial}/{label}/{minute}/{ch_name}'):
                                clk_freq = int(clk_freq_str)
                                resonator_freqs = os.listdir(
                                    f'{trials_folder}/{trial}/{label}/{minute}/{ch_name}/{clk_freq}')
                                resonator_freqs = sorted(resonator_freqs, key=resonator_fname_to_freq)
                                for f0_str in resonator_freqs:
                                    f0 = float(f0_str[:-4])
                                    if not (lf < f0 <= hf):
                                        continue
                                    loaded_spikes = np.load(
                                        f'{trials_folder}/{trial}/{label}/{minute}/{ch_name}/{clk_freq}/{f0_str}'
                                    )['spikes'].astype(np.int8)
                                    loaded_spikes = loaded_spikes[skip_beginning_s*clk_freq:]
                                    if int(clk_freq) != signal_fs:
                                        extended_spikes = np.zeros(signal_fs * (60 - skip_beginning_s), dtype=np.int8)
                                        extended_spikes[::int(signal_fs/clk_freq)] = loaded_spikes
                                        loaded_spikes = extended_spikes
                                    full_minute_spikes_input[f'{ch_name}-{band_name}'][f0] = loaded_spikes
                                    pbar.update(1)

                    for sample_start_index in range(signal_fs ,
                                                    signal_fs * (60 - skip_beginning_s) - spikes_in_sample,
                                                    step):
                        sample_start_time = sample_start_index / signal_fs
                        sample_spikes_input = {
                            input_name: torch.from_numpy(np.stack([
                                s[sample_start_index:sample_start_index + spikes_in_sample]
                                for s in resonators_arrays.values()
                            ])) for input_name, resonators_arrays in full_minute_spikes_input.items()
                        }
                        file_name = f'{sample_start_time + minute * 60}'
                        path = f'{output_path}/{trial}/{label}/{file_name}.npz'
                        path = Path(path)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if not path.is_file():
                            with open(path, 'wb') as pickle_file:
                                pickle.dump(sample_spikes_input, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
                            # np.savez_compressed(
                            #     file=path,
                            #     spikes=spikes[:, sample_start_index:sample_start_index + spikes_in_sample]
                            # )
                        pbar.update(1)


def is_file_exist(path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        return False
    # to check if its valid file
    try:
        loaded_spikes = np.load(path)['spikes']
        if len(loaded_spikes) == 0:
            return False
        if sum(loaded_spikes > 1) > 0:
            # convert timestamps encoding to spikes encoding
            spikes_encode = np.zeros(60 * clk_freq).astype('int8')
            if loaded_spikes[-1] >= 60 * clk_freq:
                raise ValueError(f'it seems that {path}, is not an array of spikes represented by spikes neither by '
                                 f'timestamps.')
            spikes_encode[loaded_spikes] = 1
            np.savez_compressed(
                file=path,
                spikes=spikes_encode
            )
        return True

    except BadZipFile as e:
        print(path)
        raise e


clk_resonators = {
    15360: [1.1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.88, 3.05, 3.39, 3.7, 4.12, 4.62, 5.09, 5.45, 5.87, 6.36, 6.8, 7.6, 8.6],
    153600: [10.5, 11.5, 12.8, 15.8, 16.6, 19.4, 22.0, 24.8, 28.4, 30.5, 34.7, 37.2, 40.2, 43.2, 47.7, 52.6, 57.2]
}

fs = 128
trials = [
    # 4,
    6, 7,
    # 3, 4, 5, 6, 7,
    # 10, 11,12,13,14,
    # 17,18,19,20,21,
    # 24,25,26,27, 28,
    # 31,32,33,34
]

channels = [
    # 'AF3',
    # 'O2',
    # 'F3',
    # 'FC5',
    # 'T7',
    # 'F7',
    # 'P8',
    # 'T8',
    # 'FC6',
    # 'F4',
    # 'F8',
    # 'AF4',
    # 'P7',
    'O1',
]

# n_channels = len(channels)
# # n_resonators = len(sum(clk_resonators.values(), start=[]))
# n_resonators = len(sum(clk_resonators.values(), []))
#
# minutes_range = {
#     'focus': [3,4,5,6,7,8,9],
#     'unfocus': [13,14,15,16,17,18,19],
#     'drowsed': [23, 24,25,26,27,28,29],
# }
#
# print(channels)
# total_minutes = sum(map(len, minutes_range.values()))
# with tqdm(total=n_channels * len(trials) * n_resonators * total_minutes) as pbar:
#     for trial in trials:
#         data = get_trial_data(trial)
#         for ch_i, ch in enumerate(channels):
#             ch_data = data[ch].values
#             # Take only first 30 minutes.
#             ch_data = ch_data[:fs * (60 * 30)]
#             ch_data /= np.max(np.abs(ch_data))
#
#             for clk_freq, resonators in clk_resonators.items():
#                 for f_i, f0 in enumerate(resonators):
#
#                     resonator = trained_resonator(
#                         freq0=float(f0),
#                         filters_folder='filters4_xi0'
#                     )
#
#                     resonator.log_out_spikes(-1)
#                     output_neuron = resonator.layers_neurons[-1].neurons[-1]
#                     # resonator.input_full_data(np.zeros(resonator.clk_freq * 5))
#                     # output_neuron.forget_logs()
#                     # minute by minute input the data.
#                     # signal_freq = [1.3, 5.1, 8.6, 16.6, 34.7]
#                     signal_freq = [3.5, 7.6, 12.5, 28, 55]
#                     for label, labeled_minutes_range in minutes_range.items():
#                         for m in labeled_minutes_range:
#                             output_folder = f'../datasets/EEG_data_for_Mental_Attention_State_Detection/EEG_spikes/{trial}/{label}/{m}/{ch}/{clk_freq}/{f0}.npz'
#                             if not is_file_exist(output_folder):
#                                 data_resampled = resample_signal(clk_freq, fs, ch_data[fs * m * 60: fs * (m + 1) * 60])
#
#                                 # x = np.linspace(0, 60, int(clk_freq * 60))
#                                 # t = x * 2 * np.pi * signal_freq[0]
#                                 # sine_wave = np.sin(t)
#                                 # for f in signal_freq[1:]:
#                                 #     t = x * 2 * np.pi * f
#                                 #     sine_wave += np.sin(t)
#                                 # sine_wave /= np.max(sine_wave)
#                                 #
#                                 # data_resampled = sine_wave
#                                 resonator.input_full_data(data_resampled)
#                                 save_output(resonator, output_folder)
#                                 output_neuron.forget_logs()
#                             pbar.set_description(f"T{trial}, Ch {ch} - {f0}, M{m}")
#                             pbar.update()

trials = [4, 5, 6, 7]
create_datasets(time_of_sample_s=.25,
                overlap=0,
                trials=trials,
                signal_fs=153600,
                output_path='../datasets/EEG_data_for_Mental_Attention_State_Detection/train_test_dataset',
                skip_beginning_s=5,
                )
