from collections import OrderedDict

import numpy as np
from numba import int32, float32

from helpers import jitclass


spec = OrderedDict([
    ('synapses_weights', float32[:]),
    ('P', float32[:]),
    ('pre_spikes_time', int32[:]),
    ('post_spikes_time', int32),
    ('A_LTP', float32),
    ('A_LTD', float32),
    ('A_tau', float32),
    ('M', float32),
    ('learning_window', int32),
    ('time', int32),
])


@jitclass(spec)
class STDP:
    """
    This code is basically implemented by this tutorial
    https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial4.html#section-2-implementation-of-stdp
    """
    def __init__(self,
                 synapses_weights: np.ndarray,
                 A_LTP: float,
                 A_LTD: float,
                 tau: float,
                 learning_window: int = 50):
        self.A_LTP = A_LTP
        self.A_LTD = A_LTD
        self.tau = tau
        self.time = 0

        self.synapses_weights = synapses_weights
        # P for every pre-synaptic
        self.P = np.zeros(synapses_weights.shape)
        self.M = 0.0
        self.pre_spikes_time = np.zeros(synapses_weights.shape) * -learning_window
        self.post_synapse_time = -learning_window
        self.learning_window = learning_window // 2

    def tick(self, pre_spikes, post_spike):
        self.time += 1
        self.P = max(-(1 / self.tau) * self.P + self.A_LTP * pre_spikes, 0)
        self.M = min((1 / self.tau) * self.M - self.A_LTD * post_spike, 0)
        if post_spike == 1:
            self.post_synapse_time = self.time
            dt = np.zeros(self.P.shape)
            # only if post spike comes after pre spike in window of self.learning_window
            dt[self.post_synapse_time - self.pre_spikes_time > self.learning_window] = 1

            # Add LTP
            self.synapses_weights += self.P * self.synapses_weights * dt

        # if there was a spike on pre synapse
        if np.any(pre_spikes) > 0:
            self.pre_spikes_time[pre_spikes == 1] = self.time
            dt = np.zeros(self.P.shape)
            # only if pre spike comes after post spike in window of self.learning_window
            dt[self.pre_spikes_time - self.post_synapse_time > self.learning_window] = 1

            # M is always negative so actually its decreasing
            self.synapses_weights += self.M * self.synapses_weights
        return self.synapses_weights



