from collections import OrderedDict

import numpy as np
from numba import int32, float32, float64

from helpers import jitclass


spec = OrderedDict([
    ('synapses_weights', float64[:]),
    ('P', float64[:]),
    ('A_LTP', float32),
    ('A_LTD', float32),
    ('tau', float32),
    ('M', float64),
    ('dt', float32),
    ('wmax', float32),
    ('wmin', float32),
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
                 f_clk: int,
                 wmax: float,
                 wmin: float,
                 ):
        """
        :param synapses_weights:
        :param A_LTP:
        :param A_LTD:
        :param tau: in seconds
        """
        self.A_LTP = A_LTP
        self.A_LTD = A_LTD
        self.dt = 1 / f_clk
        self.tau = tau
        self.wmax = wmax
        self.wmin = wmin

        self.synapses_weights = synapses_weights
        # P for every pre-synaptic
        self.P = np.zeros(synapses_weights.shape, dtype=np.float64)
        self.M = 0.0

    def tick(self, pre_spikes, post_spike):
        dp = -(self.dt / self.tau) * self.P
        dm = -(self.dt / self.tau) * self.M
        dW_post = np.minimum(pre_spikes * self.synapses_weights * self.M, 1)
        dW_pre = np.minimum(post_spike * self.synapses_weights * self.P, 1)
        self.P = np.maximum(np.minimum(dp + self.P + self.A_LTP * pre_spikes, 1.0), 0.0)
        self.M = max(min(dm + self.M + self.A_LTD * post_spike, 1.0), 0.0)
        self.synapses_weights += dW_pre - dW_post
        self.synapses_weights = np.clip(self.synapses_weights, self.wmin, self.wmax)
        return self.synapses_weights



