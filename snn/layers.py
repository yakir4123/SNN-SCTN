from collections import OrderedDict

from helpers import jitclass, numbaListType, numbaList
from snn.spiking_neuron import InputNeuron, SCTNeuron


@jitclass(OrderedDict([
    ('neurons', numbaListType(InputNeuron.class_type.instance_type)),
]))
class InputLayer:

    def __init__(self, n_neurons):
        self.neurons = numbaList([InputNeuron() for i in range(n_neurons)])


@jitclass(OrderedDict([
    ('neurons', numbaListType(SCTNeuron.class_type.instance_type)),
]))
class SCTNLayer:

    def __init__(self, neurons=None):
        if neurons is not None:
            self.neurons = numbaList(neurons)
