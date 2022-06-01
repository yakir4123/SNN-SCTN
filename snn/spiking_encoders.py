import numpy as np
from numba import njit


@njit
def generate_sine_wave(sine_size, clk_freq=1536000, amplitude=1000, zoom=200000, phase0=0):
    sine_wave = (np.arange(sine_size) / zoom + phase0)
    sine_wave = sine_wave * 2 * np.pi / clk_freq
    sine_wave = np.cumsum(sine_wave)  # phase
    return np.floor(np.sin(sine_wave) * amplitude)


@njit
def BSA_encoder(_input, threshold):
    _filter = [8, 16, 26, 35, 44, 52, 59, 64, 65, 64, 61, 57, 52, 45, 37, 29, 21, 13, 7, 4]
    output = np.zeros(len(_input))
    for i in range(len(_input)):
        error1 = 0
        error2 = 0
        for j in range(len(_filter)):
            if i + j - 1 <= len(_input):
                if _input[i + j - 1] < _filter[j]:
                    error1 += abs(_input[i + j - 1] - _filter[j])
                    error2 += abs(_input[i + j - 1])
        if error1 <= (error2 - threshold):
            output[i] = 1
            for j in range(len(_filter)):
                if i + j - 1 <= len(_input):
                    _input[i + j - 1] -= _filter[j]
        else:
            output[i] = 0
    return output

@njit
def BSA_decoder(spike_train):
    _filter = [8, 16, 26, 35, 44, 52, 59, 64, 65, 64, 61, 57, 52, 45, 37, 29, 21, 13, 7, 4]
    output = np.zeros(len(spike_train))
    spikes_length = len(spike_train)
    filter_length = len(_filter)
    for t in range(spikes_length - filter_length):
        if spike_train[t] == 1:
            for k in range(filter_length):
                output[t + k] += _filter[k]
    return output
