import json
import os
import pickle
import time
import wave
from distutils.dir_util import copy_tree
from functools import wraps
from typing import List, Dict

# import librosa
import scipy as sp
from scipy.interpolate import interp1d

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter1d

debug = False

if not debug:
    from numba.experimental import jitclass
    from numba import njit
    from numba.typed import List as numbaList
    from numba.core.types import ListType as numbaListType
    from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
    import warnings

    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

else:

    def njit(f):
        return f


    def jitclass(*args, **kwargs):
        def decorated_class(original_class):
            class dummy:
                def __init__(dummy_self):
                    dummy_self.instance_type = original_class

            original_class.class_type = dummy()
            return original_class

        return decorated_class


    numbaList = lambda _list: _list
    numbaListType = lambda _type: List[_type]


def timing(f, return_res=True, return_time=False):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'func:{f.__name__} args:({args}, {kw}] took: {te - ts:2.4f} sec')
        if return_res and return_time:
            return result, te - ts
        if return_res:
            return result
        if return_time:
            return te - ts

    return wrap


def denoise_small_values(arr, window):
    h_window = window // 2  # Half window size
    return maximum_filter1d(arr, size=window)[h_window::h_window]


def load_audio_data(audio_path, clk_freq,
                    normalize=True,
                    resample_time_ms=0,
                    remove_silence=False):
    gen_audio_path = f'sound{np.random.randint(10000)}.wav'
    with open(audio_path, "rb") as inp_f:
        data = inp_f.read()
        with wave.open(gen_audio_path, "wb") as out_f:
            out_f.setnchannels(1)
            out_f.setsampwidth(2)  # number of bytes
            out_f.setframerate(16000)
            out_f.writeframesraw(data)
    data, sr = librosa.load(gen_audio_path, sr=16000)
    os.remove(gen_audio_path)

    if normalize:
        data /= np.max(np.abs(data))

    if remove_silence:
        cumsum_data = np.abs(data).cumsum()
        window = 100
        cumsum_data[window:] = cumsum_data[window:] - cumsum_data[:-window]
        cumsum_data /= window
        th = np.std(np.abs(data)) / 3
        data = data[cumsum_data > th]

    if resample_time_ms > 0:
        freq = int((resample_time_ms / 1000) / (len(data) / 16000) * clk_freq)
    else:
        freq = clk_freq
    data = librosa.resample(data, orig_sr=sr, target_sr=freq, res_type='linear')

    return data


def copy_filter_hp_to_other_clock_frequency(clk_old, clk_new):
    clk_new_dirname = f'../filters/clk_{clk_new}/parameters'
    copy_tree(f"../filters/clk_{clk_old}/parameters", clk_new_dirname)
    scale_factor = clk_new / clk_old
    for fname in os.listdir(clk_new_dirname):
        new_filter = int(fname[2:].split('.')[0]) * scale_factor

        new_filter_file_name = f'{clk_new_dirname}/f_{new_filter:.3f}.json'
        os.rename(f'{clk_new_dirname}/{fname}', new_filter_file_name)

        with open(new_filter_file_name, 'r') as f:
            filter_parameters = json.load(f)
            filter_parameters['f0'] = new_filter
            filter_parameters['f_resonator'] *= scale_factor

        with open(new_filter_file_name, 'w') as f:
            json.dump(filter_parameters, f)


def generate_filter(*args, **kwargs):
    return generate_sinc_filter(*args, **kwargs)


def generate_sinc_filter(f0: float, start_freq: float, spectrum: float, points: int, lobe_wide: float):
    x = np.linspace(start_freq, start_freq + spectrum, points)
    x -= f0
    x /= lobe_wide / np.pi
    sinc = np.abs(np.sin(x) / x)
    sinc[np.isnan(sinc)] = 1
    return sinc


def oversample(filter_array, npts):
    interpolated = interp1d(np.arange(len(filter_array)), filter_array, axis=0, fill_value='extrapolate')
    oversampled = interpolated(np.linspace(0, len(filter_array), npts))
    return oversampled


def printable_weights(weights: np.ndarray):
    weights = weights / np.max(np.abs(weights)) * 8
    weights = np.floor(weights, dtype=np.object)
    blocks = {
        0: ' ',
        1: '\u2581',
        2: '\u2582',
        3: '\u2583',
        4: '\u2584',
        5: '\u2585',
        6: '\u2586',
        7: '\u2587',
        8: '\u2588',
    }
    colors = np.sign(weights, dtype=np.object)
    colors[colors == 0] = ''
    colors[colors == 1] = '\033[92m'
    colors[colors == -1] = '\033[91m'

    res = [f'{c}{blocks[w]}' for c, w in zip(colors, np.abs(weights))]
    res = ''.join(res) + '\033[91m'
    return res


def neurons_labels(neurons, encoder: Dict, spikes=None):
    if spikes is None:
        return ''.join([
            f'[{encoder[neuron.label]}]'
            for neuron in neurons
        ])

    return '\n'.join([
        f'[{encoder[neuron.label]}: {int(spikes[i])}]'
        for i, neuron in enumerate(neurons)
    ])


def save_network_weights(network, path):
    weights = {
        neuron._id: neuron.synapses_weights
        for neuron in network.neurons
    }
    labels = {
        neuron._id: neuron.label
        for neuron in network.neurons
    }
    with open(path, 'wb') as handle:
        pickle.dump((weights, labels), handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_network_weights(network, path):
    with open(path, 'rb') as handle:
        (weights, labels) = pickle.load(handle)

    for neuron in network.neurons:
        neuron.label = labels[neuron._id]
        neuron.synapses_weights = weights[neuron._id]
