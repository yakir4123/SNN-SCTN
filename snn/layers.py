from collections import OrderedDict

from helpers import jitclass, numbaListType, numbaList
from snn.spiking_neuron import SCTNeuron


@jitclass(OrderedDict([
    ('neurons', numbaListType(SCTNeuron.class_type.instance_type)),
]))
class SCTNLayer:

    def __init__(self, neurons=None):
        if neurons is not None:
            self.neurons = numbaList(neurons)

    def merge(self, layer):
        for neuron in layer.neurons:
            self.neurons.append(neuron)
