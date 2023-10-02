## 使う前にBindsNETをpipからインストールしてね

import numpy as np
import torch
import torch.nn.functional as F
import bindsnet

class PositionedDiehlAndCookNodes(bindsnet.network.nodes.DiehlAndCookNodes):
  def __init__(
      self,
      self.position: torch.Tensor = [0,0,0]
  ):
    super().__init__()

  def move(self,x):
    self.position = x

class PositionedInput(bindsnet.network.nodes.Input):
  def __init__(
      self,
      self.position: torch.Tensor = [0,0,0]
  ):
    super().__init__()

  def move(self,x):
    self.position = x

def compute_distance_weight(self, pos1, pos2,MAX_Distance = MAX_Distance):
    # 2つの位置座標間の距離を計算する
    tmp = np.linalg.norm(np.array(pos1) - np.array(pos2))
    if tmp > MAX_Distance:
      return 0.0
    else:
      return MAX_Distance - tmp / MAX_Distance

class PositionWeightConnection(bindsnet.network.topology.Connection):
  def __init__(
      self,
      weight_decay: float = compute_distance_weight,
      MAX_Distance: float = 10.0
  ):
    super().__init__()

    w = kwargs.get("w", None)
    if w is None:
      if (self.wmin == -np.inf).any() or (self.wmax == np.inf).any():
        w = torch.clamp(torch.rand(source.n, target.n), self.wmin, self.wmax) * compute_distance_weight(source.position, target.position)
      else:
        w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin * compute_distance_weight(source.position, target.position))
    else:
      if (self.wmin != -np.inf).any() or (self.wmax != np.inf).any():
        w = torch.clamp(torch.as_tensor(w), self.wmin, self.wmax) * compute_distance_weight(source.position, target.position)

    self.w = Parameter(w, requires_grad=False)

    b = kwargs.get("b", None)
    if b is not None:
      self.b = Parameter(b, requires_grad=False)
    else:
      self.b = None

    if isinstance(self.target, CSRMNodes):
      self.s_w = None
