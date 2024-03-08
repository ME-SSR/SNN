import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gradio as gr
import json
import traceback

# このAIはGUI上でネットワークの形状を確認、
# また視覚的に脊椎に与えるデータの宛先ニューロンを指定したり
# 出力に使う宛先ニューロンを指定したりできます。
# 与えられたデータの種類やその宛先ニューロン、
# 出力の宛先ニューロンはコンソールニューロンに対して情報が渡されます。
# コードの可視化にはgradioを使用します

TOTAL_OUTPUT_SIZE = 70

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
        self.levels = levels  # 量子化レベルの数
        self.membrane_potential = 0.0

    def forward(self, input_spike):
        self.membrane_potentials += torch.matmul(input_spike, self.weights)
        quantized_spikes = torch.floor(self.membrane_potentials / self.threshold * self.levels) / self.levels
        self.membrane_potentials *= (self.membrane_potentials < self.threshold)
        return quantized_spikes

# レートエンコーダー
class RateEncoder(nn.Module):
    def __init__(self, input_size, max_rate=10, levels=3):
        super(RateEncoder, self).__init__()
        self.input_size = input_size
        self.max_rate = max_rate
        self.levels = levels

    def forward(self, input_signal):
        spike_rates = torch.clamp(input_signal, 0, 1) * self.max_rate
        quantized_rates = torch.floor(spike_rates / self.max_rate * self.levels) / self.levels
        return quantized_rates

# 量子化スパイクニューロンを用いたネットワーク層の定義
class QuantizedSpikingLayer(nn.Module):
    def __init__(self, input_size, output_size, threshold=1.0, levels=3):
        super(QuantizedSpikingLayer, self).__init__()
        self.threshold = threshold
        self.levels = levels
        self.input_size = input_size  # この値を適切な入力サイズに設定
        self.output_size = output_size
        self.weights = nn.Parameter(torch.randn(self.input_size, self.output_size) * 0.1)  # 形状が [input_size, output_size] の重み

    def forward(self, input_spike):
        # input_spikeの形状を確認します。
        print(f"Input spike shape: {input_spike.shape}")  # デバッグ情報

        # membrane_potentials の初期化
        if not hasattr(self, 'membrane_potentials'):
            self.membrane_potentials = torch.zeros(input_spike.size(0), self.output_size, device=input_spike.device)

        # input_spikeが正しい形状を持っていることを確認（期待される形状は [1, self.input_size]）
        if input_spike.dim() == 2 and input_spike.shape[1] != self.input_size:
            raise ValueError(f"Input spike has incorrect shape. Expected [1, {self.input_size}], got {input_spike.shape}")

        # 行列乗算
        self.membrane_potentials += torch.matmul(input_spike, self.weights)

        # Quantize the spikes based on the membrane potentials
        quantized_spikes = torch.floor(self.membrane_potentials / self.threshold * self.levels) / self.levels
        # Reset the membrane potentials where they have exceeded the threshold
        self.membrane_potentials *= (self.membrane_potentials < self.threshold)
        return quantized_spikes

# シナプス
class Synapse(nn.Module):
    def __init__(self, pre_size, post_size):
        super(Synapse, self).__init__()
        self.weights = nn.Parameter(torch.randn(pre_size, post_size) * 0.1)

    def forward(self, pre_spike):
        post_input = torch.matmul(pre_spike, self.weights)
        return post_input

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
    def __init__(self, input_size, output_size, feedback_size):
        super(SpinalCord, self).__init__()
        layers = []
        for _ in range(6):  # 6層のネットワーク
            layers.append(QuantizedSpikingLayer(input_size=input_size + feedback_size, output_size=output_size, threshold=1.0, levels=3))
            input_size = output_size  # 次の層の入力サイズを更新
        self.layers = nn.ModuleList(layers)
        self.feedback_input = torch.zeros(feedback_size)

    def forward(self, TOTAL_INPUT_SIZE, higher_feedback):
        x = torch.cat([TOTAL_INPUT_SIZE, self.feedback_input + higher_feedback], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x

# 小脳アナログのネットワーク定義
class Cerebellum(nn.Module):
    def __init__(self, input_size, output_size, feedback_size):
        super(Cerebellum, self).__init__()
        layers = []
        for _ in range(6):  # 6層のネットワーク
            layers.append(QuantizedSpikingLayer(input_size=input_size + feedback_size, output_size=output_size, threshold=1.0, levels=3))
            input_size = output_size  # 次の層の入力サイズを更新
        self.layers = nn.ModuleList(layers)
        self.feedback_input = torch.zeros(feedback_size)

    def forward(self, spinal_signal, higher_feedback):
        x = torch.cat([spinal_signal, self.feedback_input + higher_feedback], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x

# 大脳アナログのネットワーク定義
class Cerebrum(nn.Module):
    def __init__(self, input_size, output_size, feedback_size):
        super(Cerebrum, self).__init__()
        layers = []
        for _ in range(6):  # 6層のネットワーク
            layers.append(QuantizedSpikingLayer(input_size=input_size + feedback_size, output_size=output_size, threshold=1.0, levels=3))
            input_size = output_size  # 次の層の入力サイズを更新
        self.layers = nn.ModuleList(layers)
        self.feedback_input = torch.zeros(feedback_size)

    def forward(self, cerebellum_signal, higher_feedback):
        x = torch.cat([cerebellum_signal, self.feedback_input + higher_feedback], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x

# 前頭葉アナログのネットワーク定義

class PrefrontalCortex(nn.Module):
    def __init__(self, input_size, output_size, feedback_size):
        super(PrefrontalCortex, self).__init__()
        layers = []
        for _ in range(6):  # 6層のネットワーク
            layers.append(QuantizedSpikingLayer(input_size=input_size + feedback_size, output_size=output_size, threshold=1.0, levels=3))
            input_size = output_size  # 次の層の入力サイズを更新
        self.layers = nn.ModuleList(layers)
        self.feedback_input = torch.zeros(feedback_size)

    def forward(self, cerebrum_signal, higher_feedback):
        x = torch.cat([cerebrum_signal, self.feedback_input + higher_feedback], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x

# タイムステップベースの処理関数を更新
def timestep_processing(TOTAL_INPUT_SIZE, spinal_cord, cerebellum, cerebrum, prefrontal_cortex, feedback_size):
    # 初期フィードバックはゼロ
    higher_feedback = torch.zeros(1, feedback_size)

    # 各層を通じて処理を実行
    spinal_output = spinal_cord(TOTAL_INPUT_SIZE, higher_feedback)
    cerebellum_feedback = cerebellum(spinal_output, higher_feedback)
    cerebrum_feedback = cerebrum(cerebellum_feedback, higher_feedback)
    prefrontal_feedback = prefrontal_cortex(cerebrum_feedback, higher_feedback)

    # フィードバックの更新
    spinal_cord.feedback_input = prefrontal_feedback  # 脊髄へのフィードバックは前頭葉から
    cerebellum.feedback_input = spinal_output  # 小脳へのフィードバックは脊髄の出力を使用
    cerebrum.feedback_input = cerebellum_feedback  # 大脳へのフィードバックは小脳の出力を使用
    prefrontal_cortex.feedback_input = cerebrum_feedback  # 前頭葉へのフィードバックは大脳の出力を使用

    return spinal_output, cerebellum_feedback, cerebrum_feedback, prefrontal_feedback

class MultiTaskSNN(nn.Module):
    def __init__(self, input_size=50, feedback_size=20, time_steps=10):
        super(MultiTaskSNN, self).__init__()
        self.input_size = input_size
        self.feedback_size = feedback_size
        self.time_steps = time_steps
        self.task_output_sizes = {}  # 空の辞書で初期化
        self.dummy_size = 0  # ダミーニューロンの数、後で更新
        self.update_network()

    def update_network(self):
        total_task_output_size = sum(self.task_output_sizes.values())
        self.dummy_size = max(0, TOTAL_OUTPUT_SIZE - total_task_output_size)  # 余ったニューロンをダミーとして計算
        total_output_size = total_task_output_size + self.dummy_size  # 実際のタスク出力とダミー出力の合計
        self.shared_layer = QuantizedSpikingLayer(input_size=self.input_size + self.feedback_size, output_size=20, threshold=1.0, levels=3)
        self.common_output_layer = QuantizedSpikingLayer(input_size=20, output_size=total_output_size, threshold=1.0, levels=3)
        self.feedback_memory = torch.zeros(self.feedback_size)

    def add_task(self, task_name, output_size):
        self.task_output_sizes[task_name] = output_size
        self.update_network()

    def remove_task(self, task_name):
        if task_name in self.task_output_sizes:
            del self.task_output_sizes[task_name]
            self.update_network()

    def forward(self, x, task):
        # 確認する: taskが有効な値である
        if task not in self.task_output_sizes:
            raise ValueError(f"Invalid task: {task}. Valid tasks are: {list(self.task_output_sizes.keys())}")

        outputs = []
        for t in range(self.time_steps):
            # 同じ修正を繰り返します: テンソルの次元を確認
            x = x.unsqueeze(0) if x.dim() == 1 else x
            feedback_input = self.feedback_memory.unsqueeze(0) if self.feedback_memory.dim() == 1 else self.feedback_memory
            combined_input = torch.cat([x, feedback_input], dim=1)
            x_shared = self.shared_layer(combined_input)
            x_output = self.common_output_layer(x_shared)
            # Update feedback memory for the next time step, ensure it's detached from the graph
            self.feedback_memory = x_output[:, :self.feedback_size].detach()

            # タスクに関する出力の選択: タスク名に基づくエラーを防ぐためのチェック追加
            start_index = sum(self.task_output_sizes[t] for t in sorted(self.task_output_sizes) if t is not None and t < task)
            end_index = start_index + self.task_output_sizes[task]
            task_output = x_output[:, start_index:end_index]
            outputs.append(task_output)
        # Return the outputs of the last time step
        return outputs[-1]

#note: エラートレースバック機能を消してはいけない！
def dynamic_snn_interface(settings):
    try:
        # JSON文字列を辞書に変換
        settings_dict = json.loads(settings)
        input_size = settings_dict['input_size']
        time_steps = settings_dict['time_steps']
        tasks = settings_dict['tasks']

        # SNNの初期化
        snn = MultiTaskSNN(input_size=input_size, time_steps=time_steps)

        # タスクの設定
        for task_name, output_size in tasks.items():
            snn.add_task(task_name, output_size)

        # 例示的な入力でSNNをテスト
        example_input = torch.rand(1, input_size)
        outputs = {task_name: snn(example_input, task_name).tolist() for task_name in tasks}
        return json.dumps(outputs)  # 辞書をJSON文字列に変換して返す
    except Exception as e:
        # エラーメッセージとトレースバックを返す
        error_message = f"An error occurred: {str(e)}"
        error_traceback = traceback.format_exc()  # エラートレースバックを取得
        return json.dumps({"error": error_message, "traceback": error_traceback})

iface = gr.Interface(
    fn=dynamic_snn_interface,
    inputs=[gr.Textbox(label="Settings (JSON format)")],
    outputs="text"
)
iface.launch()
