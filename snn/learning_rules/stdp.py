from collections import OrderedDict

import numpy as np
from numba import int32, float32, float64

from helpers import jitclass

spec = OrderedDict([
    ('synapses_weights', float64[:]),
    ('A_LTP', float32),
    ('A_LTD', float32),
    ('tau', float32),
    ('P', float64[:]),
    # ('M', float64[:]),
    ('M', float64),
    ('decay', float64),
    ('t_pre', int32[:]),
    ('t_post', int32),
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
        self.decay = np.exp(-1/tau)
        self.wmax = wmax
        self.wmin = wmin

        self.synapses_weights = synapses_weights

        # P for every pre-synaptic
        self.P = np.zeros(synapses_weights.shape, dtype=np.float64)
        # self.M = np.zeros(synapses_weights.shape, dtype=np.float64)
        self.M = 0.0

        # start with a very large counter (compare to dt -> dW -> 0)
        self.t_pre = (np.ones(synapses_weights.shape) * tau * 100).astype(np.int32)
        self.t_post = int(tau * 100)

    def reset_learning(self):
        self.P = np.zeros(self.P.shape, dtype=np.float64)
        # self.M = np.zeros(self.M.shape, dtype=np.float64)
        self.M = 0.0
        self.t_pre = (np.ones(self.t_pre.shape) * self.tau * 10).astype(np.int32)
        self.t_post = int(self.tau * 10)

    def _tick(self, pre_spikes, post_spike):
        self.t_pre += 1
        self.t_post += 1
        # reset counters if new spike arrive
        self.t_pre = ((1 - pre_spikes) * self.t_pre).astype(np.int32)
        self.t_post = int((1 - post_spike) * self.t_post)

        dW_post = (post_spike *
                   self.A_LTP *
                   np.exp(-(self.t_pre - self.t_post) / self.tau))
        # dW_post[pre_spikes == 1] = 0.

        dW_pre = (pre_spikes *
                  self.A_LTD *
                  np.exp((self.t_pre - self.t_post) / self.tau))
        if post_spike:
            dW_pre[:] = 0.

        self.synapses_weights += dW_post - dW_pre
        self.synapses_weights = np.clip(self.synapses_weights, self.wmin, self.wmax)
        return self.synapses_weights

    def tick(self, pre_spikes, post_spike):
        self.P = np.minimum(self.decay * self.P + self.A_LTP * pre_spikes, 1)
        dw_pre = self.P * post_spike
        dw_post = self.M * pre_spikes
        # self.M[pre_spikes == 1] = 0
        self.M = np.minimum(self.decay * self.M + self.A_LTD * post_spike, 1)
        self.synapses_weights += dw_pre - dw_post
        self.synapses_weights = np.clip(self.synapses_weights, self.wmin, self.wmax)
        return self.synapses_weights

        #
        # dp = -(self.dt / self.tau) * self.P
        # dm = -(self.dt / self.tau) * self.M
        # dW_post = np.minimum(pre_spikes * self.synapses_weights * self.M, 1)
        # dW_pre = np.minimum(post_spike * self.synapses_weights * self.P, 1)
        # self.P = np.maximum(np.minimum(dp + self.P + self.A_LTP * pre_spikes, 1.0), 0.0)
        # self.M = max(min(dm + self.M + self.A_LTD * post_spike, 1.0), 0.0)
        # self.synapses_weights += dW_pre - dW_post
        # self.synapses_weights = np.clip(self.synapses_weights, self.wmin, self.wmax)
        # return self.synapses_weights




