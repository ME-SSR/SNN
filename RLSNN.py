#SNNと銘打っているくせにニューロンもシナプスも仕事をしない謎のプログラム<=それSNNじゃないやないか～い(汗)
#実行しても多分エラーが出るか、動いても出てくるグラフが同じ形をしている。すなわち、ニューロンがスパイクを作らずに単に信号を横流ししているだけ。

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random


SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# CUDAの場合は以下も追加する
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Constants for SNN behavior
TIME_CONST = 5.0
SPIKE_THRESHOLD = 0.5
decay = 0.5
num_compartments = 10
DT = 0.1
ALPHA_PRE = 0.005
ALPHA_POST = 0.005
TAU_PRE = 20.0
TAU_POST = 20.0
WEIGHT_MIN = 0.0
WEIGHT_MAX = 1.0
STDP_A_PLUS = 0.005
STDP_A_MINUS = 0.005
neurotransmitter_threshold = 5.0
distance_threshold = 10
encoder_threshold = 0.5

class MultiCompartmentLIFNeuron(nn.Module):
    def __init__(self, compartments, threshold=SPIKE_THRESHOLD, decay=decay, position = (0,0,0),neurotransmitter_threshold= neurotransmitter_threshold , inter_compartment_resistance=0.5):
        super(MultiCompartmentLIFNeuron, self).__init__()
        self.compartments = compartments
        self.threshold = threshold
        self.decay = decay
        self.inter_compartment_resistance = inter_compartment_resistance
        self.voltages = [None for _ in range(self.compartments)]
        self.neurotransmitter_effects = torch.ones(self.compartments-1)  # Represents the effect of neurotransmitters between compartments
        self.neurotransmitter = Neurotransmitter()
        self.neurotransmitter_threshold = neurotransmitter_threshold
        self.trophic_factor_released = False
        self.position = torch.tensor(position, dtype=torch.float)

    def forward(self, input, neurotransmitter_signals=None):
        expanded_input = torch.zeros(*input.shape[:-1], input.shape[-1] * self.compartments).to(input.device)
        expanded_input[..., :input.shape[-1]] = input

        if neurotransmitter_signals:
            self.neurotransmitter_effects = neurotransmitter_signals

        for idx in range(self.compartments):
            if self.voltages[idx] is None:
                self.voltages[idx] = torch.zeros_like(expanded_input[..., idx*input.shape[-1]:(idx+1)*input.shape[-1]])

        for idx in range(1, self.compartments):
            delta_v = self.voltages[idx] - self.voltages[idx-1]
            transfer = delta_v * self.inter_compartment_resistance * self.neurotransmitter_effects[idx-1]
            self.voltages[idx] -= transfer
            self.voltages[idx-1] += transfer

        spikes = []
        for idx in range(self.compartments):
            # 1. 膜電位の更新
            self.voltages[idx] = self.voltages[idx] * self.decay + expanded_input[..., idx*input.shape[-1]:(idx+1)*input.shape[-1]]
            spike = (self.voltages[idx] > self.threshold).float()
            spikes.append(spike)

            # 2. 膜電位のリセット
            self.voltages[idx] = self.voltages[idx] * (1.0 - spike)

        spike = (self.voltages[-1] > self.threshold).float()
        if self.neurotransmitter.concentration > self.neurotransmitter_threshold:
            self.trophic_factor_released = True
        else:
            self.trophic_factor_released = False
        return spike
    def spike(self):
        #Release neurotransmitter when neuron fires
        self.neurotransmitter.release(0.1)


    def move(self, dx=0, dy=0, dz=0):
        # Update the position of the neuron based on the specified delta values for x, y, and z
        self.position += torch.tensor([dx, dy, dz], dtype=torch.float)

#マルチコンパートメントモデルの構造化
#フィードフォワード
class FeedForwardCompartmentLIFNeuron(MultiCompartmentLIFNeuron):
    pass  # このモデルはMultiCompartmentLIFNeuronと基本的に同じなので、特別な変更は不要です。

#リカレントネットワーク
class RecurrentCompartmentLIFNeuron(MultiCompartmentLIFNeuron):
    def forward(self, input, neurotransmitter_signals=None):
        spikes = super().forward(input, neurotransmitter_signals)

        # 前のコンパートメントからのフィードバックを処理します。
        for idx in range(1, self.compartments):
            feedback = self.voltages[idx-1] * self.neurotransmitter_effects[idx-1]
            self.voltages[idx] += feedback

        return spikes

#マルチヘッド注意機構

#注意機構

#CNN

#End of structurization



#NeuroTransmitter
class Neurotransmitter:
    def __init__(self, initial_concentration=0.0, max_concentration=10.0):
        self.concentration = initial_concentration
        self.max_concentration = max_concentration

    def release(self, amount):
        """Release neurotransmitter into the shared environment"""
        self.concentration += amount
        if self.concentration > self.max_concentration:
            self.concentration = self.max_concentration

    def consume(self, amount):
        """Consume neurotransmitter from the shared environment"""
        self.concentration -= amount
        if self.concentration < 0.0:
            self.concentration = 0.0
#End of neurotransmitter

#Conpute Distance
def compute_distance(neuron1, neuron2):
    #Compute Euclidean distance between two neurons based on their positions.
    return torch.norm(neuron1.position - neuron2.position)

def create_connections(neurons, distance_threshold=distance_threshold, max_weight=WEIGHT_MAX, min_weight=WEIGHT_MIN):
    #Create connections between neurons based on distance threshold.
    connections = {}
    for i, neuron_a in enumerate(neurons):
        for j, neuron_b in enumerate(neurons):
            if i != j:  # Don't connect a neuron to itself
                distance = compute_distance(neuron_a, neuron_b)
                if distance < distance_threshold:
                    # Here, we're making the connection strength inversely proportional to distance
                    # i.e., closer neurons have stronger connections
                    weight = F.relu(1.0 - distance / distance_threshold) * (max_weight - min_weight) + min_weight
                    connections[(i, j)] = weight
    return connections
#End of compute distance



# Define the STDP Learning Rule
class STDPLearning(nn.Module):
    def __init__(self, a_plus=0.005, a_minus=0.005, tau_plus=20.0, tau_minus=20.0):
        super(STDPLearning, self).__init__()
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.pre_trace = None
        self.post_trace = None

    def forward(self, pre_spike, post_spike):
        if self.pre_trace is None:
            self.pre_trace = torch.zeros_like(pre_spike)
        if self.post_trace is None:
            self.post_trace = torch.zeros_like(post_spike)

        self.pre_trace = self.pre_trace * (1.0 - 1.0/self.tau_plus) + pre_spike
        self.post_trace = self.post_trace * (1.0 - 1.0/self.tau_minus) + post_spike
        delta_w = self.a_plus * self.post_trace * pre_spike - self.a_minus * self.pre_trace * post_spike
        return delta_w



#Network_Modules
class SynapticConnection(nn.Module):
    def __init__(self, pre_layer, post_layer, stdp_rule, initial_weight=0.5):
        """
        pre_layer: 前の層 (spike-producing)
        post_layer: 後の層 (spike-receiving)
        stdp_rule: STDP learning rule instance
        initial_weight: シナプスの初期重み
        """
        super(SynapticConnection, self).__init__()

        self.pre_layer = pre_layer
        self.post_layer = post_layer
        self.stdp = stdp_rule

        # Initialize synaptic weights
        self.weights = nn.Parameter(torch.full((len(post_layer.neurons), len(pre_layer.neurons)), initial_weight))

    def forward(self, spikes):
        """
        Propagate the spikes through the synaptic connections.
        """
        # Compute the postsynaptic spikes by multiplying the presynaptic spikes with the weights
        postsynaptic_spikes = spikes @ self.weights.t()
        return postsynaptic_spikes

    def update(self, pre_spikes, post_spikes):
        """
        Update the synaptic weights based on the STDP learning rule.
        """
        delta_w = self.stdp(pre_spikes, post_spikes)
        self.weights.data += delta_w
        self.weights.data.clamp_(WEIGHT_MIN, WEIGHT_MAX)  # Ensure the weights remain within a valid range



def initialize_neuron_positions_3d_with_base(base_position, num_neurons, grid_dim=None):
    """
    Initialize neuron positions in a 2D grid format, but in a 3D space, based on a base position.

    Args:
    - base_position: Base (x, y, z) position for the layer.
    - num_neurons: Total number of neurons in the layer.
    - grid_dim: Dimensions of the grid. If None, the function computes a roughly square grid.

    Returns:
    - List of (x, y, z) positions for each neuron.
    """
    if grid_dim is None:
        grid_size = int(math.sqrt(num_neurons))
        grid_dim = (grid_size, grid_size)

    positions = []

    # Calculate the spacing based on the grid dimensions
    x_spacing = 1.0 / (grid_dim[0] - 1)
    y_spacing = 1.0 / (grid_dim[1] - 1)

    for i in range(grid_dim[0]):
        for j in range(grid_dim[1]):
            x = base_position[0] + i * x_spacing
            y = base_position[1] + j * y_spacing
            z = base_position[2]
            positions.append((x, y, z))

    return positions[:num_neurons]

class FeedForwardCompartmentLIFLayer(nn.Module):
    def __init__(self, num_neurons, compartments, base_position=(0.0, 0.0, 0.0), **kwargs):
        super(FeedForwardCompartmentLIFLayer, self).__init__()
        positions = initialize_neuron_positions_3d_with_base(base_position, num_neurons)
        self.neurons = nn.ModuleList([FeedForwardCompartmentLIFNeuron(compartments, position=pos, **kwargs) for pos in positions])

    def forward(self, inputs, neurotransmitter_signals=None):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron(inputs, neurotransmitter_signals))
        return torch.stack(outputs, dim=-1)


class RecurrentCompartmentLIFLayer(nn.Module):
    def __init__(self, num_neurons, compartments, base_position=(0.0, 0.0, 0.0), **kwargs):
        super(RecurrentCompartmentLIFLayer, self).__init__()
        positions = initialize_neuron_positions_3d_with_base(base_position, num_neurons)
        self.neurons = nn.ModuleList([RecurrentCompartmentLIFNeuron(compartments, position=pos, **kwargs) for pos in positions])

    def forward(self, inputs, neurotransmitter_signals=None):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron(inputs, neurotransmitter_signals))
        return torch.stack(outputs, dim=-1)

class InputLayer(nn.Module):
    def __init__(self, num_neurons, base_position, encoder=None, **kwargs):
        super(InputLayer, self).__init__()
        positions = initialize_neuron_positions_3d_with_base(base_position, num_neurons)
        self.neurons = nn.ModuleList([FeedForwardCompartmentLIFNeuron(position=pos,compartments = 1, **kwargs) for pos in positions])
        self.encoder = encoder

    def forward(self, digital_signal):
        if self.encoder:
            spikes = self.encoder(digital_signal)
        else:
            spikes = digital_signal  # Assume the input is already in spike format if no encoder is provided
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron(spikes))
        return torch.stack(outputs, dim=-1)

class OutputLayer(nn.Module):
    def __init__(self, num_neurons, base_position, decoder=None, **kwargs):
        super(OutputLayer, self).__init__()
        positions = initialize_neuron_positions_3d_with_base(base_position, num_neurons)
        self.neurons = nn.ModuleList([FeedForwardCompartmentLIFNeuron(position=pos, compartments = 1, **kwargs) for pos in positions])
        self.decoder = decoder

    def forward(self, spike_signal):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron(spike_signal))
        spikes_out = torch.stack(outputs, dim=-1)
        if self.decoder:
            return self.decoder(spikes_out)
        else:
            return spikes_out  # Return spike format if no decoder is provided

#Neurotransmitter
class BaseNeurotransmitter:
    def __init__(self, max_concentration=10.0):
        self.concentration = max_concentration / 2  # 初期濃度を半分に設定
        self.max_concentration = max_concentration

    def release(self, amount):
        """神経伝達物質を放出する"""
        self.concentration += amount
        if self.concentration > self.max_concentration:
            self.concentration = self.max_concentration

    def consume(self, amount):
        """神経伝達物質を消費する"""
        self.concentration -= amount
        if self.concentration < 0.0:
            self.concentration = 0.0

    def is_inhibitory(self):
        """神経伝達物質の濃度が半分以下の場合、抑制の作用を持つかどうかを判断する"""
        return self.concentration <= (self.max_concentration / 2)


class L_Neurotransmitter(BaseNeurotransmitter):
    pass  # 特別な機能や属性は現時点では追加しない


class R_Neurotransmitter(BaseNeurotransmitter):
    pass  # 特別な機能や属性は現時点では追加しない


#Superviased
class ErrorModulatedSTDP(STDPLearning):
    def __init__(self, error_factor=0.1, *args, **kwargs):
        super(ErrorModulatedSTDP, self).__init__(*args, **kwargs)
        self.error_factor = error_factor

    def forward(self, pre_spike, post_spike, error):
        delta_w = super().forward(pre_spike, post_spike)
        modulated_delta_w = delta_w * (1 + self.error_factor * error)
        return modulated_delta_w

#Reinforce
class RewardModulatedSTDP(STDPLearning):
    def __init__(self, reward_factor=0.1, *args, **kwargs):
        super(RewardModulatedSTDP, self).__init__(*args, **kwargs)
        self.reward_factor = reward_factor

    def forward(self, pre_spike, post_spike, reward):
        delta_w = super().forward(pre_spike, post_spike)
        modulated_delta_w = delta_w * (1 + self.reward_factor * reward)
        return modulated_delta_w

#Encoder
class RateEncoder(nn.Module):
    def __init__(self, threshold=encoder_threshold):
        super(RateEncoder, self).__init__()
        self.threshold = threshold

    def forward(self, digital_signal):
        # スパイク確率を計算し、確率が閾値を超える場合にスパイクを生成する
        spike_prob = torch.rand_like(digital_signal)
        spikes = (spike_prob < (digital_signal * self.threshold)).float()
        return spikes

#Decoder
class RateDecoder(nn.Module):
    def __init__(self, threshold=encoder_threshold):
        super(RateDecoder, self).__init__()
        self.threshold = threshold

    def forward(self, spike_signal, duration=DT):
        # 与えられた期間中のスパイクの割合を計算し、閾値を超える場合にデジタル信号を生成する
        spike_rate = spike_signal.sum() / duration
        digital_signal = (spike_rate > self.threshold).float()
        return digital_signal

import matplotlib.pyplot as plt
class SimpleSNN(nn.Module):
    def __init__(self):
      super(SimpleSNN, self).__init__()
      self.input_layer = InputLayer(num_neurons=25, base_position = (0.0, 0.0, 0.0) ,encoder=encoder)
      self.output_layer = OutputLayer(num_neurons=25, base_position = (0.0, 0.0, 3.0) , decoder=decoder)
      self.layer1_3d_base = FeedForwardCompartmentLIFLayer(num_neurons=25, compartments=5, base_position=(0.0, 0.0, 1.0))
      self.layer2_3d_base = FeedForwardCompartmentLIFLayer(num_neurons=25, compartments=5, base_position=(0.0, 0.0, 2.0))
      self.stdp_rule = STDPLearning()

      self.connection1 = SynapticConnection(self.input_layer,self.layer1_3d_base, self.stdp_rule)
      self.connection2 = SynapticConnection(self.layer1_3d_base, self.layer2_3d_base, self.stdp_rule)
      self.connection3 = SynapticConnection(self.layer2_3d_base, self.output_layer, self.stdp_rule)
    def forward(self, x, neurotransmitter_signals=None):
        x = self.connection1(x)
        plt.plot(x.detach())
        plt.show()
        x = self.connection2(x)
        plt.plot(x.detach())
        plt.show()
        return self.connection3(x)


encoder = RateEncoder()
decoder = RateDecoder()
error_modulated_stdp = ErrorModulatedSTDP()
neurotransmitter = Neurotransmitter()
model = SimpleSNN()

digital_input = torch.rand(25,25)  # Some random digital input
plt.plot(digital_input)
plt.show()
spike_signal = model(digital_input)
plt.plot(spike_signal.detach())
plt.show()
