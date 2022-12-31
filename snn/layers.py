import numba
import numpy as np

from numba import float32, types, jit
from numba.typed import Dict
from collections import OrderedDict
from snn.spiking_neuron import SCTNeuron
from helpers import jitclass, numbaListType, numbaList


@jitclass(OrderedDict([
    ('neurons', numbaListType(SCTNeuron.class_type.instance_type)),
]))
class SCTNLayer:

    def __init__(self, neurons=None):
        if neurons is not None:
            self.neurons = numbaList(neurons)

    def concat(self, layer):
        for neuron in layer.neurons:
            self.neurons.append(neuron)

    def remove_irrelevant_neurons(self):
        should_be_removed = self.should_remove_duplicates()
        should_be_removed += self.should_remove_zero_weights()
        return should_be_removed

    def should_remove_zero_weights(self):
        should_be_removed = [0 for _ in range(0)]
        for neuron in self.neurons:
            positive_weights = neuron.synapses_weights[neuron.synapses_weights > 0]
            # The synapses are irrelevant this neuron should be removed
            # No post spikes -> no learning
            if np.sum(positive_weights) < 1:
                should_be_removed.append(neuron._id)

        self.neurons = numbaList([
            neuron
            for neuron in self.neurons
            if neuron._id not in should_be_removed
        ])
        return should_be_removed

    # This code doesn't consider lf, lp and theta..
    def should_remove_duplicates(self):
        sum_of_synapses = [
            (neuron, np.sum(neuron.synapses_weights))
            for neuron in self.neurons
        ]
        sum_of_synapses = list(sorted(sum_of_synapses, key=lambda item: item[1]))
        pairs_of_neurons = zip(sum_of_synapses[:-1], sum_of_synapses[1:])

        should_be_removed = [n2._id
                             for (n1, _), (n2, _) in pairs_of_neurons
                             if len(n1.synapses_weights) == len(n2.synapses_weights)
                             and np.sum(np.abs(n1.synapses_weights - n2.synapses_weights)) < 1e-3 * len(n1.synapses_weights)]

        self.neurons = numbaList([
            neuron
            for neuron in self.neurons
            if neuron._id not in should_be_removed
        ])
        return should_be_removed

    # This code not working right now with numba..
    # For now use a simpler version above that not check for lf, lp and theta
    def __should_remove_duplicates(self):
        lf_lp_theta_groups = {}
        for neuron in self.neurons:
            # numba makes difficult life with floats and str, so instead of:
            # `group = f'{neuron.leakage_factor}_{neuron.leakage_period}_{neuron.theta}'`
            # ill do fishy thing
            group = (1 + neuron.leakage_factor * neuron.leakage_period - neuron.theta) / (
                    neuron.leakage_factor + neuron.leakage_period - 1
            )
            if group not in lf_lp_theta_groups:
                lf_lp_theta_groups[group] = []  # [self.neurons[0] for _ in range(0)]
            # lf_lp_theta_groups[group].append(neuron)

        should_be_removed = [0 for _ in range(0)]
        for neurons_group in lf_lp_theta_groups.values():
            sum_of_synapses = [
                (neuron, np.sum(neuron.synapses_weights))
                for neuron in neurons_group
            ]
            sum_of_synapses = list(sorted(sum_of_synapses, key=lambda item: item[1]))
            pairs_of_neurons = zip(sum_of_synapses[:-1], sum_of_synapses[1:])

            should_be_removed += [n2._id
                                  for (n1, w1), (n2, w2) in pairs_of_neurons
                                  if len(w1) == len(w2) and np.sum(np.abs(w1 - w2)) < 1e-3 * len(w1)]

        self.neurons = [
            neuron
            for neuron in self.neurons
            if neuron._id not in should_be_removed
        ]
        return should_be_removed
