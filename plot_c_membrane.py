import numpy as np
import json
import matplotlib.pyplot as plt

freq0 = 5000
start_freq = 0
test_size = 10_000_000
step = 1 / 1000
spikes_amount = np.zeros(test_size)


def trim_line(line):
    while line[-1] == ' ' or line[-1] == ',' or line[-1] == '\n':
        line = line[:-1]
    return line


for neuron_i in [4, 5]:#range(1, 18):
    i = 0
    try:
        with open(f'.membrane_potential/membrane_{neuron_i}.txt') as f:
            for line in f.readlines():
                line = trim_line(line)
                a = np.array(json.loads('[' + line + ']'))
                spikes_amount[i: i + len(a)] = a
                i += len(a)

        skip = 10
        cut_first = 0
        cut_end = test_size
        y = spikes_amount[cut_first:cut_end:skip]
        x = np.arange(start_freq, start_freq + test_size * step, step * skip)[cut_first:len(y) + cut_first]
        plt.plot(x, y)
        plt.axvline(x=freq0, c='red')
        plt.title(f'C - neuron {neuron_i}')
        plt.show()
    except:
        pass
