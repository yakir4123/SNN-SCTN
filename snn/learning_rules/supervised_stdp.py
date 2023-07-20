from collections import OrderedDict

import numpy as np
from numba import int32, int64, float32, float64, boolean

from utils import jitclass

spec = OrderedDict([
    ('synapses_weights', float64[:]),
    ('A', float32),
    ('tau', float32),
    ('P', float64[:]),
    ('decay', float64),
    ('dt', float32),
    ('wmax', float32),
    ('wmin', float32),
    ('is_active', boolean),
    ('desired_output', int64[:]),
])


@jitclass(spec)
class SupervisedSTDP:
    """
    This code is basically implemented by this tutorial
    https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial4.html#section-2-implementation-of-stdp
    """

    def __init__(self,
                 synapses_weights: np.ndarray,
                 A: float,
                 tau: float,
                 f_clk: int,
                 wmax: float,
                 wmin: float,
                 desired_output: np.ndarray,
                 ):
        """
        :param synapses_weights:
        :param A_LTP:
        :param A_LTD:
        :param tau: in seconds
        """
        self.A = A
        self.dt = 1 / f_clk
        self.tau = tau
        self.decay = np.exp(-1/tau)
        self.wmax = wmax
        self.wmin = wmin
        self.desired_output = desired_output

        self.is_active = False
        # P for every pre-synaptic
        self.P = np.zeros(synapses_weights.shape, dtype=np.float64)

    def reset_learning(self):
        self.P = np.zeros(self.P.shape, dtype=np.float64)

    def tick(self, synapses_weights, pre_spikes, post_spike, index):
        self.P = np.minimum(self.decay * self.P + self.A * pre_spikes, 1)
        if not self.is_active:
            if index == self.desired_output[0]:
                self.is_active = True
            else:
                return synapses_weights

        # unwanted spike
        if post_spike == 1 and index not in self.desired_output:
            synapses_weights -= self.P

        # no spike were emitted
        if post_spike == 0 and index in self.desired_output:
            synapses_weights += self.P

        synapses_weights = np.clip(synapses_weights, self.wmin, self.wmax)
        if index == self.desired_output[-1]:
            self.is_active = False
        return synapses_weights



