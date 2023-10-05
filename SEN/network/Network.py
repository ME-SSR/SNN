## 使う前にBindsNETをgitからインストールしてね

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
import bindsnet
from bindsnet.network.nodes import CSRMNodes, Nodes
from typing import Iterable, Sequence, Tuple, Optional, Union
import matplotlib.pyplot as plt
import networkx as nx

MAX_Distance = 10.0
InputShape = [10,10]

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
    def __init__(self,dt:float , batch_size:int , num_neurons: int,shape:int, radius: float, u_depth: float):
        """
        Initialize the U-Shaped Pipe topology.

        :param num_neurons: Total number of neurons along the U-shape.
        :param radius: Radius of the U-shape.
        :param u_depth: Depth of the U-shape (distance from top to bottom).
        """
        self.num_neurons = num_neurons
        self.radius = radius
        self.u_depth = u_depth
        self.shape = shape

        # Calculate the arc length
        self.arc_length = np.pi * self.radius

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
        step = (self.arc_length + self.u_depth) / self.num_neurons

        for i in range(self.num_neurons):
            if i * step < self.arc_length:  # Complete U arc
                theta = np.pi - (i * step) / self.radius
                x = self.radius * np.cos(theta)
                y = self.radius * np.sin(theta)
                positions.append((x, y, 0))
            else:  # Bottom straight line
                x = 0
                y = -2 * self.radius + (i * step - self.arc_length)
                positions.append((x, y, 0))

        return positions

    def build(self):
        positions = self._calculate_positions()
      

        # Create nodes at calculated positions
        for i, pos in enumerate(positions):
            layer_name = f"Neuron_{i}"
            if i == 0:
                layer = PositionedInput(shape=InputShape, position=torch.Tensor(pos))
                self.network.add_layer(layer, name=layer_name)
            else:
                layer = PositionedDiehlAndCookNodes(shape=self.shape, position=torch.Tensor(pos))
                self.network.add_layer(layer, name=layer_name)

            # Connect to the previous neuron
            if i > 0:
                prev_layer_name = f"Neuron_{i-1}"
                conn = PositionWeightConnection(source=self.network.layers[prev_layer_name], target=layer)
                self.network.add_connection(conn, source=prev_layer_name, target=layer_name)

        return self.network


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
