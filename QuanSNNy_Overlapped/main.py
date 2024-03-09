import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import json
import traceback
import networkx as nx
import io
from PIL import Image


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

class UBrainArchitecture(nn.Module):
    def __init__(self, input_size, output_size, feedback_size):
        super(UBrainArchitecture, self).__init__()
        # Initialize the modules
        self.spinal_cord = SpinalCord(input_size, output_size, feedback_size)
        self.cerebellum = Cerebellum(input_size, output_size, feedback_size)
        self.cerebrum = Cerebrum(input_size, output_size, feedback_size)
        self.prefrontal_cortex = PrefrontalCortex(input_size, output_size, feedback_size)

    def forward(self, total_input_size):
        # Process through the UBrain Architecture
        spinal_output, cerebellum_feedback, cerebrum_feedback, prefrontal_feedback = timestep_processing(
            total_input_size,
            self.spinal_cord,
            self.cerebellum,
            self.cerebrum,
            self.prefrontal_cortex,
            feedback_size=total_input_size.shape[1]  # Assuming feedback_size is the same as the input size for simplicity
        )
        return spinal_output, cerebellum_feedback, cerebrum_feedback, prefrontal_feedback

    def update_feedback(self, feedback):
        # This method can be used to update feedback inputs from external sources if necessary
        self.spinal_cord.feedback_input = feedback['spinal']
        self.cerebellum.feedback_input = feedback['cerebellum']
        self.cerebrum.feedback_input = feedback['cerebrum']
        self.prefrontal_cortex.feedback_input = feedback['prefrontal']

class ConvQuantizedSNNEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, levels=3):
        super(ConvQuantizedSNNEncoder, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.threshold = 1.0  # 量子化の閾値
        self.levels = levels  # 量子化レベル

    def forward(self, x):
        x = self.conv(x)  # 畳み込みを適用
        x = F.relu(x)  # 活性化関数を適用
        # 量子化処理
        x = torch.floor(x / self.threshold * self.levels) / self.levels
        return x

class EmptyDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(EmptyDecoder, self).__init__()
        # このデコーダーは入力をそのまま出力にしますが、将来的な拡張のための基本構造を提供します。
        self.linear = nn.Linear(input_size, output_size)  # 現在は使用していませんが、将来のために含めます。

    def forward(self, x):
        # 現在は入力をそのまま返しますが、後の拡張性のためにはここに処理を追加します。
        return x



def create_and_plot_network(json_params):
    try:
        # JSONパラメータを読み込む
        config = json.loads(json_params)
        encoders = config.get('encoders', [])
        decoders = config.get('decoders', [])

        if not encoders or not decoders:
            raise ValueError("JSONファイルには、'encoders' および 'decoders' セクションが必要です。")

        # ネットワークトポロジーを作成
        G = nx.DiGraph()
        for i, encoder in enumerate(encoders):
            G.add_node(f"Encoder{i+1}", type=encoder.get('type', 'Unknown'), dims=encoder.get('dims', 'Unknown'))
        for i, decoder in enumerate(decoders):
            G.add_node(f"Decoder{i+1}", type=decoder.get('type', 'Unknown'), dims=decoder.get('dims', 'Unknown'))
        for i in range(len(encoders)):
            for j in range(len(decoders)):
                G.add_edge(f"Encoder{i+1}", f"Decoder{j+1}")

        # トポロジーをプロット
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        labels = {node: f"{data['type']}({data['dims']})" for node, data in G.nodes(data=True)}
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color='lightblue')
        plt.title("Network Topology")

        # プロットをバイナリストリームとして保存し、PIL Imageに変換
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        plt.close()  # プロットをクリア

        # PIL Imageを返す
        return image

    except json.JSONDecodeError:
        return "エラー: JSONファイルの形式が無効です。"
    except ValueError as e:
        return f"エラー: {e}"
    except Exception as e:
        return f"エラー: 不明なエラーが発生しました。{e}"

# Gradioインターフェースの設定
iface = gr.Interface(
    fn=create_and_plot_network,
    inputs=[gr.Textbox(label="JSONファイルのパラメータ", placeholder="ここにJSONパラメータを入力")],
    outputs=[gr.Image(label="ネットワークトポロジー")],
    allow_flagging="never"
)

# Gradioインターフェースを起動
iface.launch()
