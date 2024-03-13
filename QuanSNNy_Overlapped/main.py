import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import json
import traceback
#import gradio as gr

import pandas as pd

class PatchedTensor:
    @staticmethod
    def extract_patches(input_tensor, patch_size, stride):
        # Ensure that the stride equals patch_size to prevent overlapping
        if stride != patch_size:
            raise ValueError("Stride must be equal to patch size to prevent overlapping.")

        # Ensure input tensor is in the expected format
        if len(input_tensor.size()) != 2:  # Assuming [H, W]
            raise ValueError("Input tensor must be of shape [H, W].")

        H, W = input_tensor.size()

        # Calculate necessary padding for H and W dimensions
        pad_height = (patch_size - H % patch_size) % patch_size
        pad_width = (patch_size - W % patch_size) % patch_size

        # Apply padding to the input tensor
        padded_input = F.pad(input_tensor, (0, pad_width, 0, pad_height), mode='constant', value=0)

        # Unfold the padded tensor to extract patches
        patches = padded_input.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
        num_patches = (H + pad_height) // patch_size * (W + pad_width) // patch_size
        patches = patches.contiguous().view(num_patches, patch_size, patch_size)
        return patches

    @staticmethod
    def merge_patches(patches, output_size):
        # Ensure output_size is in the expected format
        if len(output_size) != 2:  # Assuming [H, W]
            raise ValueError("Output size must be [H, W].")

        H, W = output_size
        output_tensor = torch.zeros(H, W, device=patches.device)
        num_patches_height = (H + patches.size(1) - 1) // patches.size(1)
        num_patches_width = (W + patches.size(2) - 1) // patches.size(2)

        for i in range(num_patches_height):
            for j in range(num_patches_width):
                output_tensor[i * patches.size(1):min((i + 1) * patches.size(1), H),
                              j * patches.size(2):min((j + 1) * patches.size(2), W)] = patches[i * num_patches_width + j, :min(patches.size(1), H - i * patches.size(1)), :min(patches.size(2), W - j * patches.size(2))]

        return output_tensor



# ■■    ■■
# ■■■   ■■
# ■ ■   ■■   ■■■■  ■   ■■  ■ ■■  ■■■■   ■ ■■■
# ■  ■  ■■  ■■  ■  ■   ■■  ■■   ■■  ■■  ■■  ■■
# ■  ■■ ■■  ■   ■■ ■   ■■  ■■   ■    ■  ■    ■
# ■   ■ ■■  ■■■■■■ ■   ■■  ■    ■    ■  ■    ■
# ■    ■■■  ■      ■   ■■  ■    ■    ■  ■    ■
# ■    ■■■  ■■     ■■  ■■  ■    ■■  ■■  ■    ■
# ■     ■■   ■■■■   ■■■■■  ■     ■■■■   ■    ■


# 量子化スパイクニューロン
class QuantizedSpikingNeuron(nn.Module):
    def __init__(self, threshold=1.0, levels=3):
        super(QuantizedSpikingNeuron, self).__init__()
        self.threshold = threshold
        self.levels = levels
        self.membrane_potential = 0.0  # Corrected name

    def forward(self, input_spike):
        # Removed the reference to self.weights, as it doesn't exist in this context
        quantized_spikes = torch.floor(self.membrane_potential / self.threshold * self.levels) / self.levels
        self.membrane_potential *= (self.membrane_potential < self.threshold)  # Fixed name
        return quantized_spikes

# 量子化スパイクニューロンを用いたネットワーク層の定義
class QuantizedSpikingLayer(nn.Module):
    def __init__(self, input_size, output_size, threshold=1.0, levels=3, use_stdp=False):
        super(QuantizedSpikingLayer, self).__init__()
        self.threshold = threshold
        self.levels = levels
        self.use_stdp = use_stdp
        self.input_size = input_size
        self.output_size = output_size
        self.weights = nn.Parameter(torch.randn(self.input_size, self.output_size) * 0.1)

        # STDP specific attributes
        if self.use_stdp:
            self.eta_plus = 0.01
            self.eta_minus = 0.01
            self.tau_plus = 20.0
            self.tau_minus = 20.0
            self.last_pre_spike = -np.inf
            self.last_post_spike = -np.inf

    def forward(self, input_spike):
        # Confirm input spike shape
        if input_spike.dim() == 2 and input_spike.shape[1] != self.input_size:
            raise ValueError(f"Input spike has incorrect shape. Expected [batch_size, {self.input_size}], got {input_spike.shape}")

        # Matrix multiplication
        self.membrane_potentials = torch.matmul(input_spike, self.weights)
        # Quantize the spikes based on the membrane potentials
        quantized_spikes = torch.floor(self.membrane_potentials / self.threshold * self.levels) / self.levels

        return quantized_spikes

    def update_weights(self, pre_spike, post_spike):
        if self.use_stdp:
            delta_t = post_spike - pre_spike
            update = torch.zeros_like(self.weights)
            update[delta_t > 0] = self.eta_plus * torch.exp(-delta_t[delta_t > 0] / self.tau_plus)
            update[delta_t < 0] = -self.eta_minus * torch.exp(delta_t[delta_t < 0] / self.tau_minus)
            self.weights += update



#                  ■                                                    ■
#   ■■■■           ■       ■■    ■■                                     ■
#  ■   ■■          ■       ■■■   ■■         ■                           ■
#  ■       ■   ■■  ■■■■■   ■ ■   ■■   ■■■■ ■■■■ ■   ■■  ■■  ■■■■   ■ ■■ ■   ■
#  ■■■     ■   ■■  ■■  ■■  ■  ■  ■■  ■■  ■  ■    ■  ■■  ■  ■■  ■■  ■■   ■  ■
#    ■■■   ■   ■■  ■    ■  ■  ■■ ■■  ■   ■■ ■    ■  ■■  ■  ■    ■  ■■   ■ ■
#      ■■  ■   ■■  ■    ■  ■   ■ ■■  ■■■■■■ ■    ■ ■ ■■ ■  ■    ■  ■    ■■■
#       ■  ■   ■■  ■    ■  ■    ■■■  ■      ■    ■■■  ■■   ■    ■  ■    ■  ■
#  ■   ■■  ■■  ■■  ■■  ■■  ■    ■■■  ■■     ■■    ■■  ■■   ■■  ■■  ■    ■  ■■
#   ■■■■    ■■■■■  ■ ■■■   ■     ■■   ■■■■   ■■   ■   ■■    ■■■■   ■    ■   ■
#
#



class Subnetwork(nn.Module):
    def __init__(self, input_size, output_size, num_layers, threshold=1.0, levels=3, use_stdp=False):
        super(Subnetwork, self).__init__()
        layers = [QuantizedSpikingLayer(input_size, output_size, threshold, levels, use_stdp)]
        for _ in range(1, num_layers):
            layers.append(QuantizedSpikingLayer(output_size, output_size, threshold, levels, use_stdp))
        self.layers = nn.ModuleList(layers)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



# ■■    ■■                                     ■
# ■■■   ■■         ■                           ■
# ■ ■   ■■   ■■■■ ■■■■ ■   ■■  ■■  ■■■■   ■ ■■ ■   ■
# ■  ■  ■■  ■■  ■  ■    ■  ■■  ■  ■■  ■■  ■■   ■  ■
# ■  ■■ ■■  ■   ■■ ■    ■  ■■  ■  ■    ■  ■■   ■ ■
# ■   ■ ■■  ■■■■■■ ■    ■ ■ ■■ ■  ■    ■  ■    ■■■
# ■    ■■■  ■      ■    ■■■  ■■   ■    ■  ■    ■  ■
# ■    ■■■  ■■     ■■    ■■  ■■   ■■  ■■  ■    ■  ■■
# ■     ■■   ■■■■   ■■   ■   ■■    ■■■■   ■    ■   ■


# ネットワークモデルの定義
# 脊髄アナログのネットワーク定義
class SpinalCord(nn.Module):
    def __init__(self, anlog_size):
        super(SpinalCord, self).__init__()
        self.layers = nn.ModuleList([
            QuantizedSpikingLayer(input_size=anlog_size, output_size=anlog_size, threshold=1.0, levels=3, use_stdp=False)
            for _ in range(6)  # 6 layers of the network
        ])
        self.feedback_input = torch.zeros(anlog_size)

    def forward(self, input_signal, higher_feedback):
        # Concatenate input_signal and higher_feedback along the feature dimension
        x = input_signal + higher_feedback 
        for layer in self.layers:
            x = layer(x)
        return x

# 小脳アナログのネットワーク定義
class Cerebellum(nn.Module):
    def __init__(self, anlog_size):
        super(Cerebellum, self).__init__()
        self.layers = nn.ModuleList([
            QuantizedSpikingLayer(input_size=anlog_size, output_size=anlog_size, threshold=1.0, levels=3, use_stdp=False)
            for _ in range(6)
        ])
        self.feedback_input = torch.zeros(anlog_size)

    def forward(self, input_signal, higher_feedback):
        # Concatenate input_signal and higher_feedback along the feature dimension
        x = input_signal + higher_feedback 
        for layer in self.layers:
            x = layer(x)
        return x

# 大脳アナログのネットワーク定義
class Cerebrum(nn.Module):
    def __init__(self, anlog_size):
        super(Cerebrum, self).__init__()
        self.layers = nn.ModuleList([
            QuantizedSpikingLayer(input_size=anlog_size, output_size=anlog_size, threshold=1.0, levels=3, use_stdp=True)
            for _ in range(6)
        ])
        self.feedback_input = torch.zeros(anlog_size)

    def forward(self, input_signal, higher_feedback):
        # Concatenate input_signal and higher_feedback along the feature dimension
        x = input_signal + higher_feedback 
        for layer in self.layers:
            x = layer(x)
        return x

# 前頭葉アナログのネットワーク定義
class PrefrontalCortex(nn.Module):
    def __init__(self, anlog_size):
        super(PrefrontalCortex, self).__init__()
        self.layers = nn.ModuleList([
            QuantizedSpikingLayer(input_size=anlog_size, output_size=anlog_size, threshold=1.0, levels=3, use_stdp=True)
            for _ in range(6)
        ])

    def forward(self, input_signal):
        # No feedback mechanism for the top layer
        x = input_signal
        for layer in self.layers:
            x = layer(x)
        return x



class UModel(nn.Module):
    def __init__(self, total_output_size, patch_size):
        super(UModel, self).__init__()
        self.patch_size = patch_size
        self.total_output_size = total_output_size
        self.encoder_subnetworks = nn.ModuleList()
        self.decoder_subnetworks = nn.ModuleList()
        # Initialize UModel layers
        self.spinal_cord = SpinalCord(total_output_size)
        self.cerebellum = Cerebellum(total_output_size)
        self.cerebrum = Cerebrum(total_output_size)
        self.prefrontal_cortex = PrefrontalCortex(total_output_size)
        # Initialize feedback inputs for each analog with zeros
        self.feedback_to_spinal = torch.zeros(1, total_output_size)
        self.feedback_to_cerebellum = torch.zeros(1, total_output_size)
        self.feedback_to_cerebrum = torch.zeros(1, total_output_size)


    def add_encoder_subnetwork(self, subnetwork):
        self.encoder_subnetworks.append(subnetwork)

    def remove_encoder_subnetwork(self, index):
        del self.encoder_subnetworks[index]

    def add_decoder_subnetwork(self, subnetwork):
        self.decoder_subnetworks.append(subnetwork)

    def remove_decoder_subnetwork(self, index):
        del self.decoder_subnetworks[index]

    def forward(self, sensor_data_list):
        # Initialize feedback if not initialized or update if necessary
        if self.feedback_to_spinal is None:
            # Assuming feedback tensors are the same size as the output
            self.feedback_to_spinal = torch.zeros(sensor_data_list[0].shape[0], self.total_output_size, device=sensor_data_list[0].device)
        
        # Process each sensor data through its subnetwork
        encoded_data = []
        for i, data in enumerate(sensor_data_list):
            if i < len(self.encoder_subnetworks):
                encoded_patch = self.encoder_subnetworks[i](data)
                encoded_data.append(encoded_patch)

        # Integrate the encoded data
        integrated_input = torch.cat(encoded_data, dim=1)  # Concatenate data horizontally

        # Pass through UModel's main layers with higher feedback
        spinal_output = self.spinal_cord(integrated_input, self.feedback_to_spinal)
        # Update feedback for the next layers
        self.feedback_to_cerebellum = spinal_output.detach()  # Detach from the current computation graph if necessary
        cerebellum_output = self.cerebellum(spinal_output, self.feedback_to_cerebellum)
        self.feedback_to_cerebrum = cerebellum_output.detach()
        cerebrum_output = self.cerebrum(cerebellum_output, self.feedback_to_cerebrum)
        # No higher feedback for the prefrontal cortex, it's the top layer
        prefrontal_output = self.prefrontal_cortex(cerebrum_output)

        # Split UModel's output into segments corresponding to each subnetwork
        output_segments = torch.split(prefrontal_output, [sub.output_size for sub in self.decoder_subnetworks], dim=1)

        # Pass the split data through the decoder subnetworks
        decoded_data = []
        for i, segment in enumerate(output_segments):
            if i < len(self.decoder_subnetworks):
                decoded_patch = self.decoder_subnetworks[i](segment)
                decoded_data.append(decoded_patch)

        # Return the list of decoded data
        return decoded_data


# モデルパラメータを設定
total_output_size = 20
patch_size = 5
batch_size = 1  # バッチサイズ

# モデルのインスタンスを作成
model = UModel(total_output_size, patch_size)
model.add_encoder_subnetwork(Subnetwork(total_output_size, total_output_size, 3, 1.0, 3, False))
model.add_decoder_subnetwork(Subnetwork(total_output_size, total_output_size, 3, 1.0, 3, False))

# テスト用のサンプルデータを生成
sensor_data_list = [torch.randn(batch_size, total_output_size) for _ in range(4)]  # 仮定: 4つのセンサーデータ

# モデルを通じてテスト入力を実行
output_data_list = model(sensor_data_list)

# 結果を表示
print("Input signal shapes:", [d.shape for d in sensor_data_list])
print("Output signal shapes:", [d.shape for d in output_data_list])

