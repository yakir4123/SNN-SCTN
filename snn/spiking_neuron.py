from collections import OrderedDict

import numpy as np
from numba import int32, float32, int8, float64, int64, int16, boolean
from helpers import jitclass, njit

spec = OrderedDict([
    ('n_synapses', int32),
    ('membrane_potential', int32),
    ('_id', int32),
    ('ca', float32),
    ('theta', int16),
    ('ca_peak', float32),
    ('delta_x', float32),
    ('max_weight', float32),
    ('min_weight', float32),
    ('min_weight', float32),
    ('leakage_timer', int16),
    ('pn_generator', int32),
    ('leakage_factor', int16),
    ('leakage_period', int16),
    ('rand_gauss_var', int32),
    ('shifting_const', float32),
    ('threshold_pulse', float32),
    ('threshold_weight', float32),
    ('activation_function', int8),
    ('gaussian_rand_order', int32),
    ('threshold_potential', float32),
    ('synapses_weights', float64[:]),
    ('threshold_depression_low', float32),
    ('threshold_depression_high', float32),
    ('threshold_potentiation_low', float32),
    ('threshold_potentiation_high', float32),

    ('log_membrane_potential', boolean),
    ('membrane_potential_graph', float32[:]),
    ('log_rand_gauss_var', boolean),
    ('rand_gauss_var_graph', int32[:]),
    ('log_ca', boolean),
    ('ca_graph', float32[:]),
    ('log_out_spikes', boolean),
    ('out_spikes', int8[:]),
    ('index', int32),
])

IDENTITY = 0
BINARY = 1
SIGMOID = 2


@jitclass(spec)
class SCTNeuron:

    def __init__(self, synapses_weights, leakage_factor=0, leakage_period=1, leakage_timer=0, threshold_weight=0.5,
                 activation_function=2, ca=0, ca_peak=1, threshold_potential=3, max_weight=1, min_weight=0,
                 theta=0, shifting_const=8e-8, threshold_potentiation_high=100, threshold_potentiation_low=10,
                 threshold_depression_high=100, threshold_depression_low=10, delta_x=5e-7, threshold_pulse=0,
                 log_membrane_potential=False, log_rand_gauss_var=False, log_ca=False, log_out_spikes=False):
        synapses_weights = synapses_weights.astype(np.float64)
        self.n_synapses = len(synapses_weights)
        self.membrane_potential = 0

        self._id = -1
        self.ca = ca
        self.theta = theta
        self.ca_peak = ca_peak
        self.delta_x = delta_x
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.leakage_timer = leakage_timer
        self.shifting_const = shifting_const
        self.leakage_factor = leakage_factor
        self.leakage_period = leakage_period
        self.threshold_pulse = threshold_pulse
        self.threshold_weight = threshold_weight
        self.threshold_potential = threshold_potential
        self.synapses_weights = np.copy(synapses_weights)
        self.threshold_depression_low = threshold_depression_low
        self.threshold_depression_high = threshold_depression_high
        self.threshold_potentiation_low = threshold_potentiation_low
        self.threshold_potentiation_high = threshold_potentiation_high

        self.rand_gauss_var = 0
        self.gaussian_rand_order = 8
        self.pn_generator = 1
        self.activation_function = activation_function

        self.log_membrane_potential = log_membrane_potential
        self.log_ca = log_ca
        self.log_rand_gauss_var = log_rand_gauss_var
        self.log_out_spikes = log_out_spikes
        self.membrane_potential_graph = np.zeros(100).astype('float32')
        self.ca_graph = np.zeros(100).astype('float32')
        self.out_spikes = np.zeros(100).astype('int8')
        self.rand_gauss_var_graph = np.zeros(100).astype('int32')
        self.index = 0

    def ctn_cycle(self, f, enable, learning):
        emit_spike = self._kernel(f, enable)
        if learning:
            self._learn(f, emit_spike)

        if self.log_membrane_potential:
            if self.index == len(self.membrane_potential_graph):
                self.membrane_potential_graph = np.concatenate((self.membrane_potential_graph,
                                                                np.zeros(self.index).astype('float32')))
            self.membrane_potential_graph[self.index] = self.membrane_potential
        if self.log_rand_gauss_var:
            if self.index == len(self.rand_gauss_var_graph):
                self.rand_gauss_var_graph = np.concatenate((self.rand_gauss_var_graph,
                                                            np.zeros(self.index).astype('int32')))
            self.rand_gauss_var_graph[self.index] = self.rand_gauss_var
        if self.log_ca:
            if self.index == len(self.ca_graph):
                self.ca_graph = np.concatenate((self.ca_graph,
                                                np.zeros(self.index).astype('float32')))
            self.ca_graph[self.index] = self.ca
        if self.log_out_spikes:
            if self.index == len(self.out_spikes):
                self.out_spikes = np.concatenate((self.out_spikes,
                                                  np.zeros(self.index).astype('int8')))
            self.out_spikes[self.index] = emit_spike
        self.index += 1
        return emit_spike

    def _kernel(self, f, enable):
        if enable:
            if self.leakage_factor < 3:
                self.membrane_potential += np.sum(np.multiply(f, self.synapses_weights))
                self.membrane_potential += self.theta
            else:
                lf = (2 ** (self.leakage_factor - 3))
                for i in range(len(self.synapses_weights)):
                    self.membrane_potential += f[i] * self.synapses_weights[i] * lf
                    self.membrane_potential = int(self.membrane_potential)
                # self.membrane_potential += int(np.sum(
                #     np.multiply(f, self.synapses_weights * (2 ** (self.leakage_factor - 3)))))
                self.membrane_potential += self.theta * (2 ** (self.leakage_factor - 3))

            self.membrane_potential = np.clip(np.array([self.membrane_potential]), -524287, 524287)[0]
            # self.membrane_potential = np.clip(np.array([self.membrane_potential]), -2147483648, 2147483647)[0]

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

    def _learn(self, f, out_pulse):
        if self.membrane_potential > self.threshold_potential and \
                self.threshold_potentiation_high > self.ca > self.threshold_potentiation_low:
            self.synapses_weights[f == 1] += self.delta_x
        elif self.membrane_potential < self.threshold_potential and \
                self.threshold_depression_high > self.ca > self.threshold_depression_low:
            self.synapses_weights[f == 1] -= self.delta_x
        self.synapses_weights += self.shifting_const
        self.synapses_weights[self.synapses_weights <= self.threshold_weight] -= 2 * self.shifting_const
        self.synapses_weights = np.clip(self.synapses_weights, self.min_weight, self.max_weight)

        if out_pulse:
            self.ca += self.ca_peak
        else:
            self.ca /= np.e

    def _activation_function_identity(self):
        const = 32767
        c = self.membrane_potential + const
        m = 65536

        if self.membrane_potential > const:
            emit_spike = 1
            self.rand_gauss_var = const
        elif self.membrane_potential < -const:
            emit_spike = 0
            self.rand_gauss_var = const
        else:
            self.rand_gauss_var = int(self.rand_gauss_var + c + 1)
            if self.rand_gauss_var >= m:
                # self.rand_gauss_var = self.rand_gauss_var % m
                self.rand_gauss_var -= m
                emit_spike = 1
            else:
                emit_spike = 0
        return emit_spike

    def _activation_function_binary(self):
        return self.membrane_potential > self.threshold_pulse

    def _activation_function_sigmoid(self):
        self.rand_gauss_var = 0
        for _ in range(self.gaussian_rand_order):
            self.rand_gauss_var += self.pn_generator & 0x1fff
            self.pn_generator = (self.pn_generator >> 1) | (
                    (self.pn_generator & 0x4000) ^ ((self.pn_generator & 0x0001) << 14))
        return self.membrane_potential > self.rand_gauss_var

    def __hash__(self):
        return self._id


@njit
def createEmptySCTN():
    return SCTNeuron(np.array([0]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, False, False)


@jitclass({'_id': int32})
class InputNeuron:

    def __init__(self):
        self._id = -1
