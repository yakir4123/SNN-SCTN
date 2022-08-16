import numpy as np
import optuna
from matplotlib import pyplot as plt

from snn.resonator import freq_of_resonator, test_frequency, CustomResonator

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
    start_freq = 0
    f_pulse = 1.536 * (10 ** 6)
    for freq0, LF, LP in learns:
        study_name = f'Study-{freq0}-{LF}-{LP}'
        storage = "sqlite:///example.db"
        study = optuna.create_study(study_name=study_name,
                                    storage=storage,
                                    load_if_exists=True)
        study.op

        # gains = study.best_params
        # spectrum = 2 * freq0
        # # step = 1 / (test_size // spectrum)
        # step = 1 / (40_000)
        # test_size = int(spectrum / step)
        #
        # # step = 1/10_000
        # my_resonator = CustomResonator(freq0, f_pulse, LF, LP, **gains)
        # # my_resonator = Resonator(freq0, f_pulse)
        # # plot_network(my_resonator.network)
        # my_resonator.network.log_membrane_potential(-1)
        # # my_resonator.network.log_out_spikes(-1)
        # test_frequency(my_resonator, start_freq=start_freq, step=step, test_size=test_size)
        # for i in [-1]:
        #     neuron = my_resonator.network.neurons[1]
        #     LF = neuron.leakage_factor
        #     LP = neuron.leakage_period
        #     neuron = my_resonator.network.neurons[i]
        #     skip = 50
        #     membrane = neuron.membrane_potential_graph[:neuron.index]
        #     y = membrane[::skip]
        #     x = np.arange(start_freq, start_freq + test_size*step, step*skip)[:len(y)]
        #     plt.plot(x, y)
        #     plt.axvline(x=freq0, c='red')
        #     f = int(freq_of_resonator(f_pulse, LF, LP))
        #     plt.title(f'neuron {i}, LF = {LF}, LP = {LP}, df = {freq0}, f = {f}')
        #     plt.show()
        # print("Nice")