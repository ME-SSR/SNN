#SNNと銘打っているくせにニューロンもシナプスも仕事をしない謎のプログラム<=それSNNじゃないやないか～い(汗)
#実行しても多分エラーが出るか、動いても出てくるグラフが同じ形をしている。すなわち、ニューロンがスパイクを作らずに単に信号を横流ししているだけ。

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
import torch.nn as nn
import bindsnet
from bindsnet.network.nodes import CSRMNodes, Nodes
from typing import Iterable, Sequence, Tuple, Optional, Union
import matplotlib.pyplot as plt
import networkx as nx


MAX_Distance = 10.0
threhold = -52.0
decay = 1


class PositionedDiehlAndCookNodes(bindsnet.network.nodes.DiehlAndCookNodes):
  def __init__(
      self,
      position: torch.Tensor = [0,0,0],
      n: Optional[int] = None,
      shape: Optional[Iterable[int]] = None,
      traces: bool = False,
      traces_additive: bool = False,
      tc_trace: Union[float, torch.Tensor] = 20.0,
      trace_scale: Union[float, torch.Tensor] = 1.0,
      sum_input: bool = False,
      **kwargs,
  ):
    self.position = position

    super().__init__(
      n=n,
      shape=shape,
      traces=traces,
      traces_additive=traces_additive,
      tc_trace=tc_trace,
      trace_scale=trace_scale,
      sum_input=sum_input,
      thresh = threhold
    )

  def move(self,x):
    self.position = self.position + x

class PositionedInput(bindsnet.network.nodes.Input):
  def __init__(
      self,
      position: torch.Tensor = [0,0,0],
      n: Optional[int] = None,
      shape: Optional[Iterable[int]] = None,
      traces: bool = False,
      traces_additive: bool = False,
      tc_trace: Union[float, torch.Tensor] = 20.0,
      trace_scale: Union[float, torch.Tensor] = 1.0,
      sum_input: bool = False,
      **kwargs,
  ):
    self.position = position
    super().__init__(
      n=n,
      shape=shape,
      traces=traces,
      traces_additive=traces_additive,
      tc_trace=tc_trace,
      trace_scale=trace_scale,
      sum_input=sum_input,
    )
  def move(self,x):
    self.position = self.position + x

def compute_distance_weight(pos1, pos2,MAX_Distance = MAX_Distance):
  # 2つの位置座標間の距離を計算する
  tmp = np.linalg.norm(np.array(pos1) - np.array(pos2))
  if tmp > MAX_Distance:
    return 0.0
  else:
    return (MAX_Distance - tmp) / MAX_Distance

class PositionWeightConnection(bindsnet.network.topology.Connection):
    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = None,
        MAX_Distance: float = MAX_Distance,
        **kwargs,
    ):
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        w = kwargs.get("w", None)
        if w is None:
            # Compute distance weights for all source-target neuron pairs
            distance_weights = torch.zeros(source.n, target.n)
            for i in range(source.n):
                for j in range(target.n):
                    distance_weights[i, j] = compute_distance_weight(
                        pos1=source.position, pos2=target.position
                    )
            if (self.wmin == -np.inf).any() or (self.wmax == np.inf).any():
                w = torch.clamp(torch.rand(source.n, target.n), self.wmin, self.wmax) * distance_weights
            else:
                w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin) * distance_weights
        else:
            distance_weight = compute_distance_weight(pos1=source.position, pos2=target.position)
            if (self.wmin != -np.inf).any() or (self.wmax != np.inf).any():
                w = torch.clamp(torch.as_tensor(w), self.wmin, self.wmax) * distance_weight

        self.w = Parameter(w, requires_grad=False)

        b = kwargs.get("b", None)
        if b is not None:
            self.b = Parameter(b, requires_grad=False)
        else:
            self.b = None

        if isinstance(self.target, CSRMNodes):
            self.s_w = None

class UShapedPipeTopology:
    def __init__(self, dt: float, batch_size: int, num_neurons: int, shape: int,
                 radius: float, u_depth: float, net_thickness: int, net_interval: int,
                 additional_neurons: int = 1):
        """
        Initialize the U-Shaped Pipe topology.

        :param num_neurons: Total number of neurons along the U-shape.
        :param radius: Radius of the U-shape.
        :param u_depth: Depth of the U-shape (distance from top to bottom).
        :param additional_neurons: Number of additional neurons at each end.
        """
        self.num_neurons = num_neurons
        self.radius = radius
        self.u_depth = u_depth
        self.shape = shape
        self.net_thickness = net_thickness
        self.net_interval = net_interval
        self.additional_neurons = additional_neurons

        self.left_end_positions = []
        self.right_end_positions = []
        self.EndOfLayer = []
        self.StartOfLayer = []

        super().__init__()

        self.dt = dt
        self.batch_size = batch_size

        # Create the network
        self.network = Network()

    def _calculate_positions(self):
        """
        Calculate the positions of neurons along the U-shaped pipe.
        """
        positions = []
        tmp = 0

        for j in range(self.net_thickness):

            # Calculate the arc length
            self.arc_length = np.pi * (self.radius + j * self.net_interval)
            step = (self.arc_length + self.u_depth) / (self.num_neurons + j * self.net_interval)
            for i in range(self.num_neurons + j * self.net_interval):
                if i * step < self.arc_length:  # Complete U arc
                    theta = np.pi - (i * step) / (self.radius + self.net_interval * j)
                    x = (self.radius + self.net_interval * j) * np.cos(theta)
                    y = (self.radius + self.net_interval * j) * np.sin(theta)
                    positions.append((x, y, 0))
                else:  # Bottom straight line
                    x = 0
                    y = -2 * self.radius + (i * step - self.arc_length)
                    positions.append((x, y, 0))
            if not j == 0:
                self.EndOfLayer.append(tmp + self.num_neurons + j * self.net_interval)
            else:
                self.EndOfLayer.append(self.num_neurons - 1)
            tmp = self.EndOfLayer[-1]
            self.StartOfLayer.append(tmp - self.num_neurons - j * self.net_interval + 1)



        for i in range(self.additional_neurons):
            # Neuron at the left end
            x_left = positions[0][0] - (i + 1) * self.net_interval
            y_left = positions[0][1]
            self.left_end_positions.append((x_left, y_left - self.net_interval, 0))

            # Neuron at the right end
            x_right = positions[-1][0] - (i + 1) * self.net_interval
            y_right = positions[-1][1]
            self.right_end_positions.append((x_right, y_right - self.net_interval, 0))

        return positions

    def build(self):
        positions = self._calculate_positions()

        layer = PositionedInput(shape=InputShape, position=torch.mean(torch.Tensor(self.left_end_positions), dim = 0))
        self.network.add_layer(layer, name = "Input")
        layer = PositionedDiehlAndCookNodes(shape = self.shape, position = torch.mean(torch.Tensor(self.right_end_positions), dim = 0))
        self.network.add_layer(layer, name = "Output")


        # Create nodes at calculated positions
        for i, pos in enumerate(positions):
            layer_name = f"Neuron_{i}"
            layer = PositionedDiehlAndCookNodes(shape=self.shape, position=torch.Tensor(pos))
            self.network.add_layer(layer, name=layer_name)

            # Connect to the previous neuron
            if i > 0:
              prev_layer_name = f"Neuron_{i-1}"
              if i in self.EndOfLayer:
                  conn = PositionWeightConnection(source=layer, target=layer, weight_decay=decay)
                  self.network.add_connection(conn, source=layer_name, target="Output")
                  self.network.add_connection(conn, source=layer_name, target=layer_name)
              elif i in self.StartOfLayer:
                  conn = PositionWeightConnection(source=layer, target=layer, weight_decay=decay)
                  self.network.add_connection(conn, source="Input", target=layer_name)
                  self.network.add_connection(conn, source=layer_name, target=layer_name)
              else:
                  conn = PositionWeightConnection(source=self.network.layers[prev_layer_name], target=layer, weight_decay=decay)
                  #forward
                  self.network.add_connection(conn, source=prev_layer_name, target=layer_name)
                  #recurrent
                  self.network.add_connection(conn, source=layer_name, target=prev_layer_name)
                  self.network.add_connection(conn, source=layer_name, target=layer_name)
            elif i == 0:
                conn = PositionWeightConnection(source=layer, target=layer, weight_decay=decay)
                self.network.add_connection(conn, source="Input", target=layer_name)
                self.network.add_connection(conn, source=layer_name, target=layer_name)

        return self.network


# オートエンコーダーの定義
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim):
        super(Autoencoder, self).__init__()

        # エンコーダー部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU()
        )

        # デコーダー部分
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 出力を [0, 1] の範囲に制限
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class SNNEncoder:
    def __init__(self, max_time, spike_rate):
        self.max_time = max_time
        self.spike_rate = spike_rate

    def encode(self, data):
        spike_data = torch.zeros((self.max_time,) + data.shape)
        for t in range(self.max_time):
            spike_data[t] = torch.rand_like(data) < data * self.spike_rate
        return spike_data

class SNNDecoder:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def decode(self, spike_data):
        firing_rate = spike_data.float().mean(dim=0)
        decoded_data = (firing_rate > self.threshold).float()
        return decoded_data

def string_to_normalized_tensor(s, max_size):
    # 文字をUnicodeコードポイントに変換
    codepoints = [ord(c) for c in s]

    # 最大のコードポイントで正規化
    normalized = [cp / 0x10FFFF for cp in codepoints]

    # max_sizeに合わせて0でパディング
    while len(normalized) < max_size:
        normalized.append(0.0)

    return torch.tensor([normalized])

def tensor_to_strings(tensor):
    tensor = tensor.squeeze()  # 余分な次元を削除
    batch_size, seq_len = tensor.shape[:2]

    strings = []
    for i in range(batch_size):
        string_batch = []
        for j in range(seq_len):
            codepoint = int((tensor[i, j] * 0x10FFFF).item())
            if codepoint != 0:  # 0のコードポイントはスキップ
                string_batch.append(chr(codepoint))
        strings.append(''.join(string_batch))

    return strings






def torch_fix_seed(seed=1):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def visualize_network(network):
    G = nx.DiGraph()

    pos = {}  # This dictionary will store the positions of the neurons

    for layer_name, layer in network.layers.items():
        G.add_node(layer_name)
        pos[layer_name] = (layer.position[0].item(), layer.position[1].item())  # Using the actual neuron position

    for conn_name, conn in network.connections.items():
        G.add_edge(conn_name[0], conn_name[1])

    # Use the actual positions for drawing
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", node_shape="s", alpha=0.5, width=2.0)
    plt.show()
