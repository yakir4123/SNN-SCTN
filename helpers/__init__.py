import time
import wave
from functools import wraps
from typing import List

import librosa
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


def load_audio_data(audio_path, clk_freq, normalize=True):
    with open(audio_path, "rb") as inp_f:
        data = inp_f.read()
        with wave.open("sound.wav", "wb") as out_f:
            out_f.setnchannels(1)
            out_f.setsampwidth(2)  # number of bytes
            out_f.setframerate(16000)
            out_f.writeframesraw(data)
    audio_path = 'sound.wav'

    data, sr = librosa.load(audio_path, sr=16000)

    data = librosa.resample(data, orig_sr=sr, target_sr=clk_freq, res_type='linear')

    if normalize:
        data /= np.max(data)
    return data


def generate_filter(*args, **kwargs):
    # return oversample(np.load('../filters/filter_2777.npy'), points=kwargs['points'])
    return generate_sinc_filter(*args, **kwargs)


def generate_sinc_filter(f0: float, start_freq: float, spectrum: float, points: int, lobe_wide: float):
    x = np.linspace(start_freq, start_freq + spectrum, points)
    x -= f0
    x /= lobe_wide/np.pi
    sinc = np.abs(np.sin(x)/x)
    sinc[np.isnan(sinc)] = 1
    return sinc


def oversample(filter_array, npts):
    interpolated = interp1d(np.arange(len(filter_array)), filter_array, axis=0, fill_value='extrapolate')
    oversampled = interpolated(np.linspace(0, len(filter_array), npts))
    return oversampled
