import optuna
import numpy as np

from helpers import denoise_small_values, skew_score, generate_filter
from snn.resonator import CustomResonator, test_frequency, OptimizationResonator
from scipy import stats


def objective(trial):
    gain_factor = 9344 / ((2 ** (2 * LF - 3)) * (1 + LP))
    min_gain = 0.2 * gain_factor
    max_gain = 2 * gain_factor
    theta_gain = (
        trial.suggest_float('th_gain0', min_gain, max_gain),
        trial.suggest_float('th_gain1', min_gain, max_gain),
        trial.suggest_float('th_gain2', min_gain, max_gain),
        trial.suggest_float('th_gain3', min_gain, max_gain)
    )
    weight_gain = (
        trial.suggest_float('weight_gain0', min_gain, max_gain),
        trial.suggest_float('weight_gain1', min_gain, max_gain),
        trial.suggest_float('weight_gain2', min_gain, max_gain),
        trial.suggest_float('weight_gain3', min_gain, max_gain),
        trial.suggest_float('weight_gain4', min_gain, max_gain)
    )
    amplitude_gain = trial.suggest_float('amplitude_gain', min_gain, max_gain)
    # weight_gain = 1.1, 0.9, 1., 1., 1.
    # theta_gain = 1., 1., 1., 1.
    # amplitude_gain = 1.
    start_freq = 0
    f_pulse = 1.536 * (10 ** 6)
    step = 1 / 40_000
    spectrum = 2 * freq0
    test_size = int(spectrum / step)
    my_resonator = OptimizationResonator(freq0,
                                         f_pulse,
                                         LF,
                                         LP,
                                         theta_gain,
                                         weight_gain,
                                         amplitude_gain)
    my_resonator.network.log_membrane_potential(-1)
    test_frequency(my_resonator, start_freq=start_freq, step=step, test_size=test_size)

    neuron = my_resonator.network.neurons[-1]
    membrane = neuron.membrane_potential_graph[:neuron.index]
    membrane = denoise_small_values(np.abs(membrane), 10000)
    membrane -= np.min(membrane)
    membrane /= np.max(membrane)
    return np.sum((f_filter - membrane) ** 2)
    # convolve = np.convolve(membrane, f_filter, mode='full')
    # convolve /= np.max(convolve)
    # argx1 = np.argmax(membrane)
    # argx2 = np.argmax(convolve)
    # x1 = np.linspace(start_freq, start_freq + spectrum, len(membrane))
    # x2 = np.linspace(start_freq, start_freq + spectrum, len(convolve))
    # return np.max(convolve) * (1 / (1 + np.abs(freq0 - x1[argx1]))) * (1 / (1 + np.abs(freq0 - x2[argx2])))
    # return stats.skew(membrane)
    # return skew_score(membrane)


if __name__ == '__main__':
    learns = [
        (104, 5, 72),
        # (2777, 3, 10),
        # (3395, 3, 8),
        # (4365, 3, 6),
        # (6111, 3, 4),
        # (5555, 2, 10),
        # (6790, 2, 8),
        # (8730, 2, 6),
        # (12223, 2, 4)
    ]
    f_filter = generate_filter('filter_104', 0)
    for freq0, LF, LP in learns:
        study_name = f'Study-{freq0}-{LF}-{LP}-sinc025pi'
        storage = "sqlite:///example.db"
        # optuna.delete_study(study_name=study_name, storage=storage)
        study = optuna.create_study(study_name=study_name,
                                    storage=storage,
                                    direction='minimize',
                                    load_if_exists=True)
        study.optimize(objective, n_trials=300)

        print(study.best_params)
