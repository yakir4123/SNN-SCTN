import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import resample
from scipy.signal import butter, filtfilt

from tqdm import tqdm
from pathlib import Path
from snn.resonator import create_excitatory_inhibitory_resonator


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
    # filename = 'eeg_record30.mat'
    mat = loadmat(data_root + filename)
    data = pd.DataFrame.from_dict(mat["o"]["data"][0, 0])

    # Limit the data to the 7 valid EEG leads.
    dat = data.filter(list(range(3, 17)))
    dat.columns = list(range(1, 15))
    dat = dat.filter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17], axis=1)
    labels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    dat.columns = labels

    # Filter the data, high pass 2 Hz, low pass 40 Hz.
    # lo, hi = 2, 40
    # Do the filtering.
    # datf = bp_filter(dat.to_numpy(), lo, hi, hz)

    # Convert back to a dataframe.
    # dat = pd.DataFrame({c: datf[:, i] for i, c in enumerate(labels)})

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


def generate_spikes(resonator, data_resampled, append_with_zeros=True):
    resonator.input_full_data(data_resampled)

def save_output(resonator, spikes_output_path):
    output_neuron = resonator.layers_neurons[-1].neurons[-1]
    np.savez_compressed(
        file=spikes_output_path,
        spikes=output_neuron.out_spikes()
    )


def create_datasets(time_of_sample_s, overlap, trials, output_path):
    trials_folder = '../datasets/EEG_data_for_Mental_Attention_State_Detection/EEG_spikes'
    signal_fs = 8415
    spikes_in_sample = signal_fs * time_of_sample_s
    step = int(spikes_in_sample * (1 - overlap))
    for trial in tqdm(trials):
        # every sample of 1s is 8Kb of memory!
        # try:
        # for ch_name in os.listdir(f'{trials_folder}/{trial}'):
        #     for clk_freq in os.listdir(f'{trials_folder}/{trial}/{ch}'):
        #         for f0 in os.listdir(f'{trials_folder}/{trial}/{ch}/{clk_freq}'):
        #             np.load(f'{trials_folder}/{trial}/{ch_name}/{clk_freq}/{f0}')['spikes'].astype(np.int8)
        # except ValueError:
        #     raise ValueError(f'Exception at {trials_folder}/{trial}/{ch_name}/{clk_freq}/{f0}')

        spikes = np.array([
            [
                np.load(f'{trials_folder}/{trial}/{ch_name}/{clk_freq}/{f0}')['spikes'].astype(np.int8)
                for clk_freq in os.listdir(f'{trials_folder}/{trial}/{ch}')
                for f0 in os.listdir(f'{trials_folder}/{trial}/{ch}/{clk_freq}')
            ]
            for ch_name in os.listdir(f'{trials_folder}/{trial}')
        ])
        for sample_start_index in range(0, spikes.shape[-1] - spikes_in_sample, step):
            sample_start_time = int(sample_start_index / signal_fs * 1000)
            file_name = f'{trial}_{sample_start_time}'
            np.savez_compressed(
                file=f'{output_path}/{file_name}',
                spikes=spikes[:, :, sample_start_time:sample_start_time + spikes_in_sample]
            )

def is_file_exist(path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.is_file()


clk_resonators = {
    76800: ['4.000', '4.500', '5.000', '5.900', '6.950'],
    153600: ['8.000', '9.000', '10.000', '11.800', '13.900'],
    307200: ['16.000', '18.000', '20.000', '23.600', '27.800'],
    614400: ['32.000', '36.000', '40.000', '47.200', '55.600'],
}

fs = 128
channels = [
    'AF3', 'F7',
    'F3', 'FC5',
    'T7', 'P7',
    'O1', 'O2',
    'P8', 'T8',
    'FC6', 'F4',
    'F8', 'AF4'
    'FC6', 'F4',
]

trails = [
    3,4,5,6,7,
        10,11,12,13,14,
        17,18,19,20,21,
        24,25,26,27,28,
        31,32,33,34
    ]

print('Find constant normalization value!')
normalization_value = -np.inf
for trial in trails:
    data = get_trial_data(trial)
    for ch_i, ch in enumerate(data.columns):
        ch_data = data[ch].values
        normalization_value = np.max([normalization_value, np.max(np.abs(ch_data))])


channels = [
    # 'FC6',
    # 'F4',
    # 'F8',
    # 'AF4'
    # 'AF3',
    # 'F7',
    # 'F3',
    # 'FC5',
    # 'T7',
    # 'P7',
    # 'O1',
    # 'O2',
    # 'P8',
    'T8',
]

n_channels = len(channels)
n_resonators = len(sum(clk_resonators.values(), start=[]))

minutes_range = {
    'focus': [3,4,5,6,7,8,9],
    'unfocus': [13,14,15,16,17,18,19],
    'drowesed': [23, 24,25,26,27,28,29],
}
print(channels)
total_minutes = sum(map(len, minutes_range.values()))
with tqdm(total=n_channels * len(trails) * n_resonators * total_minutes) as pbar:
    for trial in trails:
        data = get_trial_data(trial)
        for ch_i, ch in enumerate(channels):
            ch_data = data[ch].values
            # Take only first 30 minutes.
            ch_data = ch_data[:fs * (60 * 30)]
            ch_data /= normalization_value
            for clk_i, (clk_freq, list_of_f0) in enumerate(clk_resonators.items()):
                for f_i, f0 in enumerate(list_of_f0):
                    resonator = create_excitatory_inhibitory_resonator(
                        freq0=f0,
                        clk_freq=clk_freq)

                    resonator.log_out_spikes(-1)
                    output_neuron = resonator.layers_neurons[-1].neurons[-1]
                    resonator.input_full_data(np.zeros(resonator.clk_freq * 5))
                    output_neuron.forget_logs()
                    # minute by minute input the data.
                    for label, labeled_minutes_range in minutes_range.items():
                        for m in labeled_minutes_range:
                            output_folder = f'../datasets/EEG_data_for_Mental_Attention_State_Detection/EEG_spikes_clk/{trial}/{label}/{m}/{ch}/{clk_freq}/{f0}.npz'
                            if not is_file_exist(output_folder):
                                data_resampled = resample_signal(clk_freq, fs, ch_data[fs * m * 60: fs * (m+1) * 60])
                                resonator.input_full_data(data_resampled)
                            save_output(resonator, output_folder)
                            output_neuron.forget_logs()
                            pbar.update()

# create_datasets(3, .5, range(31, 35), '../datasets/EEG_data_for_Mental_Attention_State_Detection/train_test_dataset')
