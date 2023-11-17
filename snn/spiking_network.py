from utils import *
from snn.layers import SCTNLayer
from collections import OrderedDict
from numba import int8, int32, float32
from snn.graphs import DirectedEdgeListGraph
from snn.spiking_neuron import SCTNeuron, create_SCTN


@jitclass(OrderedDict([
    ('clk_freq', int32),
    ('clk_freq_i', int32),
    ('clk_freq_qrt', int8),
    ('amplitude', float32[:]),
    ('clk_freq_sine', float32[:]),
    ('enable_by', numbaListType(int32)),
    ('spikes_graph', DirectedEdgeListGraph.class_type.instance_type),
    ('layers_neurons', numbaListType(SCTNLayer.class_type.instance_type)),

]))
class SpikingNetwork:

    def __init__(self):
        """
        @enable_by: list of all neurons that map if neuron is enabled by other neuron
            id is enabled by self.enable_by[id]
        @spikes_graph: DirectedEdgeListGraph graph that map connections between neurons
            AND the spikes that enter the neurons
        @neurons: list of all the neurons inside the network
        @layers_neurons: list of all the layers
        """
        self.enable_by = numbaList([np.int32(0) for _ in range(0)])
        self.spikes_graph = DirectedEdgeListGraph()

        # numba needs to identify what the list type, so create empty list
        self.layers_neurons = numbaList([SCTNLayer(None) for _ in range(0)])
        self.amplitude = np.array([np.float32(0) for _ in range(0)])

    def add_amplitude(self, amplitude):
        self.amplitude = np.append(self.amplitude, np.float32(amplitude))

    def add_layer(self, layer):
        for new_neuron in layer.neurons:
            self.add_neuron(new_neuron)
            if len(self.layers_neurons) > 0:
                [self.spikes_graph.connect(neuron, new_neuron) for neuron in self.layers_neurons[-1].neurons]
        self.layers_neurons.append(layer)
        return self

    def add_neuron(self, new_neuron, layer=-1):
        # self.neurons.append(new_neuron)
        self.spikes_graph.add_node(new_neuron)
        self.enable_by.append(-1)

        if layer != -1:
            self.layers_neurons[layer].neurons.append(new_neuron)
            # connect all neurons from previous layer to this new neuron
            if layer > 0:
                [self.spikes_graph.connect(neuron, new_neuron) for neuron in self.layers_neurons[layer - 1].neurons]
            # connect this neuron to all of next layer neurons from previous layer to this new neuron
            if layer < len(self.layers_neurons) - 1:
                [self.spikes_graph.connect(new_neuron, neuron) for neuron in self.layers_neurons[layer + 1].neurons]

    def add_network(self, network):
        new_id_offset = self.spikes_graph.add_graph(network.spikes_graph)
        for neuron in network.neurons:
            neuron._id += new_id_offset
            self.neurons.append(neuron)

        for amplitude in network.amplitude:
            self.add_amplitude(amplitude)

        for i in range(len(network.enable_by)):
            if network.enable_by[i] != -1:
                network.enable_by[i] += new_id_offset
            self.enable_by.append(network.enable_by[i])

        for i in range(len(network.layers_neurons)):
            if i == len(self.layers_neurons):
                self.layers_neurons.append(network.layers_neurons[i])
            else:
                self.layers_neurons[i].concat(network.layers_neurons[i])

    def get_layer(self, i):
        return self.layers_neurons[i]

    def connect(self, source_neuron, target_neuron):
        self.connect_by_id(source_neuron._id, target_neuron._id)

    def connect_by_id(self, source_id, target_id):
        self.spikes_graph.connect_by_id(source_id, target_id)

    def connect_enable_by_id(self, source_id, target_id):
        self.enable_by[target_id] = source_id

    def remove_irrelevant_neurons(self, weak_th=0):
        should_be_removed = [
            nid
            for layer in self.layers_neurons
            for nid in layer.remove_irrelevant_neurons(weak_th)
        ]

        self.spikes_graph.remove_any_connections(should_be_removed)
        return len(should_be_removed)

    def input(self, spike_train):
        # first update that input neurons send spikes
        for i, layer in enumerate(self.layers_neurons):
            for j, neuron in enumerate(layer.neurons):
                enable = self.is_enable(neuron)
                if i == 0:
                    emit_spike = neuron.ctn_cycle(spike_train, enable)
                else:
                    emit_spike = neuron.ctn_cycle(self.spikes_graph.get_input_spikes_to(neuron), enable)
                self.spikes_graph.update_spike(neuron, emit_spike)
        last_neurons = np.array([n._id for n in self.layers_neurons[-1].neurons])
        return self.spikes_graph.spikes[last_neurons]

    def input_full_data(self, data):
        classes = np.zeros(len(self.layers_neurons[-1].neurons))
        for i, potential in enumerate(data):
            res = self.input_potential(potential)
            classes += res
        return classes

    def input_full_data_spikes(self, spike_train, stop_on_first_spike=False):
        classes = np.zeros(len(self.layers_neurons[-1].neurons))
        for i, spikes in enumerate(spike_train):
            res = self.input(spikes)
            if stop_on_first_spike and np.any(res):
                return res.astype(np.float64)
            classes += res
        return classes

    def input_potential(self, potential):
        potential = (potential * self.amplitude).astype(np.int16)

        for i, p in enumerate(potential):
            neuron = self.layers_neurons[0].neurons[i]
            neuron.membrane_potential = p

        return self.input(np.zeros(len(potential)))

    def forget_logs(self):
        for neuron in self.neurons:
            neuron.forget_logs()

    def __getitem__(self, nid):
        if nid < 0:
            neurons = list(self.neurons)
            return neurons[nid]
        for layer in self.layers_neurons:
            for neuron in layer.neurons:
                if neuron._id == nid:
                    return neuron

    @property
    def neurons(self):
        return [
            neuron
            for layer in self.layers_neurons
            for neuron in layer.neurons
        ]

    @property
    def neurons_count(self):
        return len(list(self.neurons))

    def reset_learning(self):
        for neuron in self.neurons:
            neuron.reset_learning()

    def reset_input(self):
        for neuron in self.neurons:
            neuron.index = 0
            neuron.leakage_timer = 0
            neuron.membrane_potential = 0

    def is_enable(self, neuron):
        # if nothing connected to enable gate or a there was a spike from the neuron that connected to enable gate
        return self.enable_by[neuron._id] == -1 or self.spikes_graph.spikes[self.enable_by[neuron._id]] == 1

    def log_membrane_potential(self, neurons_id):
        self[neurons_id].log_membrane_potential = True

    def log_rand_gauss_var(self, neurons_id):
        self[neurons_id].log_rand_gauss_var = True

    def log_out_spikes(self, neurons_id):
        self[neurons_id].log_out_spikes = True


def get_labels(network: SpikingNetwork):
    return np.array([n.label for n in network.layers_neurons[-1].neurons])
