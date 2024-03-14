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
        self.update_structure(input_size, output_size, num_layers, threshold, levels, use_stdp)

    def update_structure(self, input_size, output_size, num_layers, threshold=1.0, levels=3, use_stdp=False):
        self.input_size = input_size
        self.output_size = output_size
        layers = [QuantizedSpikingLayer(input_size, output_size, threshold, levels, use_stdp)]
        for _ in range(1, num_layers):
            layers.append(QuantizedSpikingLayer(output_size, output_size, threshold, levels, use_stdp))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DynamicDataHandler:
    def __init__(self):
        self.encoder_data_pairs = []
        self.decoder_data_pairs = []
        self.names = {}  # インデックスと名前のマッピング

    def add_pair(self, encoder_name, encoder_subnetwork, sensor_data, decoder_name, decoder_subnetwork, output_data):
        # 名前とサブネットワークをペアとして追加
        self.encoder_data_pairs.append((encoder_subnetwork, sensor_data))
        self.decoder_data_pairs.append((decoder_subnetwork, output_data))

    def remove_pair_by_name(self, name):
        # 名前に基づいてペアを削除
        if name in self.names:
            index = self.names[name]
            del self.encoder_data_pairs[index]
            del self.decoder_data_pairs[index]
            del self.names[name]  # 名前をマップから削除
            # インデックスを更新
            new_names = {}
            for n, i in self.names.items():
                new_names[n] = i - 1 if i > index else i
            self.names = new_names
        else:
            raise ValueError(f"Pair with name {name} does not exist.")

    def get_index(self, name):
        # 名前からインデックスを取得
        if name in self.names:
            return self.names[name]
        else:
            raise ValueError(f"Name {name} does not exist.")

    def get_encoder_by_name(self, name):
        # 名前からエンコーダを取得
        index = self.get_index(name)
        return self.encoder_data_pairs[index][0]

    def get_decoder_by_name(self, name):
        # 名前からデコーダを取得
        index = self.get_index(name)
        return self.decoder_data_pairs[index][0]

    def get_encoders(self):
        return [pair[0] for pair in self.encoder_data_pairs]

    def get_decoders(self):
        return [pair[0] for pair in self.decoder_data_pairs]

    def get_sensor_data(self):
        return [pair[1] for pair in self.encoder_data_pairs]

    def get_output_data(self):
        return [pair[1] for pair in self.decoder_data_pairs]



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


#                          ■         ■
# ■■     ■■■               ■         ■
# ■■■    ■■■               ■         ■
# ■■■    ■■■   ■■■■    ■■■■■   ■■■■  ■
# ■ ■   ■■■■  ■■  ■■  ■■  ■■  ■■  ■  ■
# ■  ■  ■ ■■  ■    ■  ■    ■  ■   ■■ ■
# ■  ■  ■ ■■  ■    ■  ■    ■  ■■■■■■ ■
# ■  ■■■  ■■  ■    ■  ■    ■  ■      ■
# ■   ■■  ■■  ■■  ■■  ■■  ■■  ■■     ■
# ■   ■   ■■   ■■■■    ■■■ ■   ■■■■  ■


class UModel(nn.Module):
    def __init__(self, total_output_size, patch_size):
        super(UModel, self).__init__()
        self.dynamic_data_handler = DynamicDataHandler()  # ここで動的データハンドラを初期化
        self.patch_size = patch_size
        self.total_output_size = total_output_size
        # メインネットワークを初期化
        self.initialize_main_networks(total_output_size)
        # UModelレイヤーを初期化
        self.spinal_cord = SpinalCord(total_output_size)
        self.cerebellum = Cerebellum(total_output_size)
        self.cerebrum = Cerebrum(total_output_size)
        self.prefrontal_cortex = PrefrontalCortex(total_output_size)
        # 各アナログのフィードバック入力をゼロで初期化
        self.feedback_to_spinal = torch.zeros(1, total_output_size)
        self.feedback_to_cerebellum = torch.zeros(1, total_output_size)
        self.feedback_to_cerebrum = torch.zeros(1, total_output_size)

    def initialize_main_networks(self, total_output_size):
        self.spinal_cord = SpinalCord(total_output_size)
        self.cerebellum = Cerebellum(total_output_size)
        self.cerebrum = Cerebrum(total_output_size)
        self.prefrontal_cortex = PrefrontalCortex(total_output_size)

    def process_sensor_data(self, sensor_data_list):
        # 処理を実行する前に、encoder_subnetworksと長さが一致するか確認
        assert len(sensor_data_list) == len(self.dynamic_data_handler.get_encoders()), \
            "Length of sensor_data_list must match the number of encoder subnetworks."

        encoded_data_list = []

        # エンコード処理
        for i, sensor_data in enumerate(sensor_data_list):
            encoder = self.dynamic_data_handler.get_encoders()[i]
            if sensor_data.size(1) > encoder.input_size:
                print(f"Sensor data {i} is bigger than encoder size, trimming to fit.")
                input_data = sensor_data[:, :encoder.input_size]
            else:
                input_data = sensor_data
            encoded_data = encoder(input_data)
            encoded_data_list.append(encoded_data)

        # エンコードされたデータを統合
        integrated_input = torch.cat(encoded_data_list, dim=1) if encoded_data_list else torch.zeros(1, self.total_output_size)
        if integrated_input.size(1) < self.total_output_size:
            padding_size = self.total_output_size - integrated_input.size(1)
            integrated_input = F.pad(integrated_input, (0, padding_size), 'constant', 0)

        # 中央処理ユニットを通す
        spinal_output = self.spinal_cord(integrated_input, self.feedback_to_spinal)
        self.feedback_to_spinal = cerebellum_output = self.cerebellum(spinal_output, self.feedback_to_cerebellum)
        self.feedback_to_cerebellum = cerebrum_output = self.cerebrum(cerebellum_output, self.feedback_to_cerebrum)
        self.feedback_to_cerebrum = prefrontal_output = self.prefrontal_cortex(cerebrum_output)

        # デコード処理
        output_data_list = []
        for decoder in self.dynamic_data_handler.get_decoders():
            segment_size = decoder.input_size
            segment = prefrontal_output[:, :segment_size]
            decoded_data = decoder(segment)
            output_data_list.append(decoded_data)

        return output_data_list




# モデルパラメータを設定
total_output_size = 20
patch_size = 5
batch_size = 1  # バッチサイズ

# UModelのインスタンスを作成
model = UModel(total_output_size, patch_size)

# 個々のエンコーダーとデコーダーのサブネットワーク、および関連するデータを名前付きで追加
# 各センサーデータとダミーの出力データ
sensor_data_camera1 = torch.randn(batch_size, 3)
sensor_data_camera2 = torch.randn(batch_size, 4)
sensor_data_camera3 = torch.randn(batch_size, 5)
sensor_data_camera4 = torch.randn(batch_size, 6)

# エンコーダーとデコーダーのサブネットワークを定義して追加
encoder_camera1 = Subnetwork(3, 5, 3, 1.0, 3, False)
encoder_camera2 = Subnetwork(4, 5, 3, 1.0, 3, False)
encoder_camera3 = Subnetwork(5, 5, 3, 1.0, 3, False)
encoder_camera4 = Subnetwork(6, 5, 3, 1.0, 3, False)

decoder_motor1 = Subnetwork(5, 5, 3, 1.0, 3, False)
decoder_motor2 = Subnetwork(5, 5, 3, 1.0, 3, False)
decoder_motor3 = Subnetwork(5, 5, 3, 1.0, 3, False)
decoder_motor4 = Subnetwork(5, 5, 3, 1.0, 3, False)

dummy_output_data = torch.empty(batch_size, 5)  # 処理前なので空のテンソルを作成

# 名前を付けてペアを追加
model.dynamic_data_handler.add_pair("カメラ1", encoder_camera1, sensor_data_camera1, "モーター1", decoder_motor1, dummy_output_data)
model.dynamic_data_handler.add_pair("カメラ2", encoder_camera2, sensor_data_camera2, "モーター2", decoder_motor2, dummy_output_data)
model.dynamic_data_handler.add_pair("カメラ3", encoder_camera3, sensor_data_camera3, "モーター3", decoder_motor3, dummy_output_data)
model.dynamic_data_handler.add_pair("カメラ4", encoder_camera4, sensor_data_camera4, "モーター4", decoder_motor4, dummy_output_data)

# センサーデータリストをモデルを通じて処理
output_data_lists = model.process_sensor_data([sensor_data_camera1, sensor_data_camera2, sensor_data_camera3, sensor_data_camera4])

# 処理後の出力データでデコーダーのデータを更新
for i, output_data in enumerate(output_data_lists):
    # デコーダーデータを更新
    model.dynamic_data_handler.decoder_data_pairs[i] = (model.dynamic_data_handler.get_decoders()[i], output_data)

print(output_data_lists)
