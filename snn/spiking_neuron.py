from collections import OrderedDict

import numpy as np
from numba import int32, float32, int8, float64, int16, boolean, optional, types
from helpers import jitclass, njit
from snn.learning_rules.stdp import STDP
spec = OrderedDict([
    ('_id', int32),
    ('theta', float32),
    ('pn_generator', int32),
    ('leakage_timer', int16),
    ('identity_const', int32),
    ('leakage_factor', int16),
    ('leakage_period', float32),
    ('rand_gauss_var', int32),
    ('threshold_pulse', float32),
    ('activation_function', int8),
    ('gaussian_rand_order', int32),
    ('membrane_potential', float32),
    ('synapses_weights', optional(float64[:])),
    ('membrane_should_reset', boolean),
    ('stdp', optional(STDP.class_type.instance_type)),

    ('index', int32),
    ('out_spikes', int8[:]),
    ('log_out_spikes', boolean),
    ('log_rand_gauss_var', boolean),
    ('label', optional(types.string)),
    ('rand_gauss_var_graph', int32[:]),
    ('log_membrane_potential', boolean),
    ('_membrane_potential_graph', float32[:]),
    ('membrane_sample_max_window', float32[:]),
])


IDENTITY = 0
BINARY = 1
SIGMOID = 2


@jitclass(spec)
class SCTNeuron:

    @njit
    def __init__(self):
        self.membrane_potential = 0.0

        self._id = -1
        self.theta = 0
        self.identity_const = 32767
        self.leakage_timer = 0
        self.leakage_factor = 0
        self.leakage_period = 0
        self.threshold_pulse = 0
        self.synapses_weights = None
        self.stdp = None
        self.label = None

        self.rand_gauss_var = 0
        self.gaussian_rand_order = 8
        self.pn_generator = 1
        self.activation_function = 0
        self.membrane_should_reset = True

        self.log_membrane_potential = False
        self.log_rand_gauss_var = False
        self.log_out_spikes = False
        self._membrane_potential_graph = np.zeros(100).astype('float32')
        self.membrane_sample_max_window = np.zeros(10000).astype('float32')
        self.out_spikes = np.zeros(100).astype('int8')
        self.rand_gauss_var_graph = np.zeros(100).astype('int32')
        self.index = 0

    def ctn_cycle(self, pre_spikes, enable):
        emit_spike = self._kernel(pre_spikes, enable)

        if self.stdp is not None:
            self.synapses_weights = self.stdp.tick(pre_spikes, emit_spike)

        if self.log_membrane_potential:
            sample_window_size = len(self.membrane_sample_max_window)
            if self.index // sample_window_size == len(self._membrane_potential_graph):
                self._membrane_potential_graph = np.concatenate((self._membrane_potential_graph,
                                                                np.zeros(self.index // sample_window_size).astype('float32')))

            self.membrane_sample_max_window[self.index % sample_window_size] = self.membrane_potential
            if self.index % sample_window_size == sample_window_size - 1:
                self.membrane_sample_max_window[np.isnan(self.membrane_sample_max_window)] = 0
                self._membrane_potential_graph[self.index // sample_window_size] = np.max(np.abs(self.membrane_sample_max_window))
        if self.log_rand_gauss_var:
            if self.index == len(self.rand_gauss_var_graph):
                self.rand_gauss_var_graph = np.concatenate((self.rand_gauss_var_graph,
                                                            np.zeros(self.index).astype('int32')))
            self.rand_gauss_var_graph[self.index] = self.rand_gauss_var
        if self.log_out_spikes:
            if self.index == len(self.out_spikes):
                self.out_spikes = np.concatenate((self.out_spikes,
                                                  np.zeros(self.index).astype('int8')))
            self.out_spikes[self.index] = emit_spike

        if self.membrane_should_reset and emit_spike:
            self.membrane_potential = 0

        self.index += 1
        return emit_spike

    def _kernel(self, f, enable):
        if enable:
            if self.leakage_factor < 3:
                self.membrane_potential += np.sum(np.multiply(f, self.synapses_weights))
                self.membrane_potential += self.theta
            else:
                lf = (2 ** (self.leakage_factor - 3))
                self.membrane_potential += np.sum(np.multiply(f, self.synapses_weights)) * lf
                self.membrane_potential += self.theta * lf

            self.membrane_potential = np.clip(np.array([self.membrane_potential]), -524287, 524287)[0]
        # can't use dictionary of function because of numba ...
        if self.activation_function == IDENTITY:
            emit_spike = self._activation_function_identity()
        elif self.activation_function == BINARY:
            emit_spike = self._activation_function_binary()
        elif self.activation_function == SIGMOID:
            emit_spike = self._activation_function_sigmoid()
        else:
            raise ValueError("Only 3 activation functions are supported [IDENTITY, BINARY, SIGMOID]")

        if enable:
            if self.leakage_timer >= self.leakage_period:
                if self.membrane_potential < 0:
                    decay_delta = (-self.membrane_potential) // (2 ** self.leakage_factor)
                else:
                    decay_delta = -(self.membrane_potential // (2 ** self.leakage_factor))
                self.membrane_potential += decay_delta
                self.leakage_timer = 0
            else:
                self.leakage_timer += 1
        return emit_spike

    def set_stdp(self, A_LTP, A_LTD, tau, clk_freq, wmax, wmin):
        self.stdp = STDP(self.synapses_weights,
                         A_LTP,
                         A_LTD,
                         tau,
                         clk_freq,
                         wmax,
                         wmin,
                         )

    def reset_learning(self):
        if self.stdp is not None:
            self.stdp.reset_learning()

    def _activation_function_identity(self):
        const = self.identity_const
        c = self.membrane_potential + const
        m = 2 * (self.identity_const + 1)

        if self.membrane_potential > const:
            emit_spike = 1
            self.rand_gauss_var = const
        elif self.membrane_potential < -const:
            emit_spike = 0
            self.rand_gauss_var = const
        else:
            self.rand_gauss_var = int(self.rand_gauss_var + c + 1)
            if self.rand_gauss_var >= m:
                self.rand_gauss_var = self.rand_gauss_var % m
                emit_spike = 1
            else:
                emit_spike = 0
        return emit_spike

    def _activation_function_binary(self):
        if self.membrane_potential > self.threshold_pulse:
            return 1
        return 0

    def _activation_function_sigmoid(self):
        self.rand_gauss_var = 0
        for _ in range(self.gaussian_rand_order):
            self.rand_gauss_var += self.pn_generator & 0x1fff
            self.pn_generator = (self.pn_generator >> 1) | (
                    (self.pn_generator & 0x4000) ^ ((self.pn_generator & 0x0001) << 14))
        if self.membrane_potential > self.rand_gauss_var:
            return 1
        return 0

    def __hash__(self):
        return self._id

    def membrane_potential_graph(self):
        return self._membrane_potential_graph[:self.index // len(self.membrane_sample_max_window)]
