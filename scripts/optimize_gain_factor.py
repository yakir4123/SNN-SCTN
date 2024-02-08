import json
from copy import copy

import yaml
import optuna
import numpy as np
from optuna.samplers import CmaEsSampler, TPESampler

from utils import denoise_small_values, generate_sinc_filter, generate_filter, oversample
from snn.resonator import test_resonator_on_chirp, BaseResonator, lf_lp_options


def objective(trial):
    LF, LP, f_resonator = _lf_lp_options[trial.suggest_int('lf_lp_option', 0, len(_lf_lp_options)-1)]
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

    spectrum = 2 * freq0
    step = 1 / 1_000
    test_size = int(spectrum / step)
    my_resonator = BaseResonator(freq0,
                                 f_pulse,
                                 LF,
                                 LP,
                                 theta_gain,
                                 weight_gain,
                                 amplitude_gain)
    my_resonator.network.log_membrane_potential(-1)
    test_resonator_on_chirp(my_resonator, start_freq=start_freq, step=step, test_size=test_size)

    neuron = my_resonator.network.neurons[-1]
    membrane = neuron.membrane_potential_graph()
    # membrane = denoise_small_values(np.abs(membrane), 10000)
    membrane -= np.min(membrane)
    max_membrane = np.max(membrane)
    main_freq_membrane = membrane[len(membrane)//2]
    if max_membrane == 0 or main_freq_membrane == 0:
        return 99999
    # membrane /= max_membrane
    membrane /= main_freq_membrane

    f_filter = generate_filter(f_resonator, start_freq=start_freq, spectrum=spectrum,
                               points=len(membrane), lobe_wide=0.125 * f_resonator)

    res = np.sum((f_filter - membrane) ** 2)
    return res


if __name__ == '__main__':
    start_freq = 0
    f_pulse = int(1.536 * (10 ** 6) * 2)
    # learns = [100 * (1.18 ** i) for i in range(25, 29)]
    # learns = [i for i in range(5, 101, 5)]
    learns = [3334]

    with open("../secret.yaml", 'r') as stream:
        secrets = yaml.safe_load(stream)

    storage = f'postgresql://{secrets["USER"]}:{secrets["PASSWORD"]}@{secrets["ENDPOINT"]}:{secrets["PORT"]}/{secrets["DBNAME"]}'
    for freq0 in learns:
        _lf_lp_options = lf_lp_options(freq0, f_pulse)
        _lf_lp_options_indices = abs(_lf_lp_options[:, 2] - freq0) / freq0 < 0.1
        _lf_lp_options = _lf_lp_options[_lf_lp_options_indices]

        study_name = f'Study{f_pulse}-{freq0}_5'
        # optuna.delete_study(study_name=study_name, storage=storage)
        study = optuna.create_study(study_name=study_name,
                                    storage=storage,
                                    sampler=TPESampler(seed=42),
                                    direction='minimize',
                                    load_if_exists=True)

        study.optimize(objective, n_trials=50)

        with open(f"../filters/clk_{f_pulse}/parameters/f_{int(freq0)}.json", 'w') as best_params_f:
            res = copy(study.best_params)
            LF, LP, f_resonator = _lf_lp_options[res['lf_lp_option']]
            res['f0'] = int(freq0)
            res['LF'] = LF
            res['LP'] = LP
            res['f_resonator'] = f_resonator
            res['th'] = 50

            json.dump(res, best_params_f, indent=4)
    print('Done!')

 # {'lf_lp_option': 2, 'th_gain0': 24.796592600235087, 'th_gain1': 8.500271828625097, 'th_gain2': 7.698360138202604, 'th_gain3': 7.739870518948241, 'weight_gain0': 10.91548614497665, 'weight_gain1': 16.71059902329521, 'weight_gain2': 14.271515089914804, 'weight_gain3': 10.573501804404541, 'weight_gain4': 18.999494073304135, 'amplitude_gain': 6.58589865793566}