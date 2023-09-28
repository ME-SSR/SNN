#RLSNN-Transformer
#今の所空間座標を持つマルチコンパートメントモデルでTransformerを構築している。
#(なお、まだ空間座標による接続強度の変化は実装していない。)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Constants for SNN behavior
TIME_CONST = 5.0
SPIKE_THRESHOLD = 0.5
num_compartments = 2
DT = 1.0
ALPHA_PRE = 0.005
ALPHA_POST = 0.005
TAU_PRE = 20.0
TAU_POST = 20.0
WEIGHT_MIN = 0.0
WEIGHT_MAX = 1.0
STDP_A_PLUS = 0.005
STDP_A_MINUS = 0.005

# Number of layers in the Transformer Encoder and Decoder
NUM_LAYERS = 6



# Define the spiking neuron with the Leaky Integrate and Fire (LIF) model

class MultiCompartmentLIFNeuron(nn.Module):
    def __init__(self, compartments, threshold=1.0, decay=0.99, inter_compartment_resistance=0.5):
        super(MultiCompartmentLIFNeuron, self).__init__()
        self.compartments = compartments
        self.threshold = threshold
        self.decay = decay
        self.inter_compartment_resistance = inter_compartment_resistance
        self.voltages = [None for _ in range(self.compartments)]
        self.neurotransmitter_effects = torch.ones(self.compartments-1)  # Represents the effect of neurotransmitters between compartments

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
            self.voltages[idx] = self.voltages[idx] * self.decay + expanded_input[..., idx*input.shape[-1]:(idx+1)*input.shape[-1]]
            spike = (self.voltages[idx] > self.threshold).float()
            spikes.append(spike)
            self.voltages[idx] = self.voltages[idx] * (1.0 - spike)

        spike = (self.voltages[-1] > self.threshold).float()
        return spike

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

#残差接続
class ResidualSTDPTransformerBlock(nn.Module):
    def __init__(self, k, heads, dropout=0.1):
        super().__init__()

        self.k = k

        # Multi-head self-attention mechanism
        self.attention = MultiHeadedAttention(heads, k, dropout=dropout)

        # Feed-forward neural network (position-wise)
        self.feed_forward = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention output
        attended = self.attention(x, x, x, mask)

        # Add residual connection
        x = self.norm1(attended + x)

        # Feed-forward network
        fed_forward = self.feed_forward(x)

        # Add residual connection
        x = self.norm2(fed_forward + x)

        return x

#マルチヘッド注意機構
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        # Create linear projections for Q, K, V
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])

        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout)

        # Final output projection
        self.out_projection = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # Check for masking
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Number of batches
        nbatches = query.size(0)

        # Project the input for Q, K, V for all heads
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # Apply the attention mechanism for all heads
        x, self.attn = self.attention(query, key, value, mask=mask)

        # Concatenate the results and apply the final linear transformation
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.out_projection(x)

#注意機構
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # Calculate the dot product attention
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

        # Apply the mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Normalize the scores with softmax
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        # Return the weighted values
        return torch.matmul(p_attn, value), p_attn

#モジュール構造
class CompartmentModule(nn.Module):
    def __init__(self, compartments, threshold=1.0, decay=0.99, inter_compartment_resistance=0.5):
        super(CompartmentModule, self).__init__()
        self.neuron = MultiCompartmentLIFNeuron(
            compartments=compartments,
            threshold=threshold,
            decay=decay,
            inter_compartment_resistance=inter_compartment_resistance
        )

    def forward(self, input, neurotransmitter_signals=None):
        return self.neuron.forward(input, neurotransmitter_signals)

#CNN
class MultiCompartmentCapsuleLayer(nn.Module):
    def __init__(self, num_compartments, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None):
        super(MultiCompartmentCapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_capsules = num_capsules
        self.num_compartments = num_compartments
        self.capsules = nn.ModuleList()

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            for _ in range(self.num_compartments):
                self.capsules.append(
                    nn.ModuleList([
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in range(num_capsules)
                    ])
                )

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = torch.zeros(*priors.size()).to(x.device)
            for i in range(self.num_capsules):
                probs = F.softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
                if i != self.num_capsules - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            all_outputs = []
            for compartment in self.capsules:
                outputs = [caps(x).view(x.size(0), -1, 1) for caps in compartment]
                outputs = torch.cat(outputs, dim=-1)
                all_outputs.append(outputs)
            outputs = torch.cat(all_outputs, dim=-1)
            outputs = self.squash(outputs)
        return outputs


class MultiCompartmentCapsuleNetwork(nn.Module):
    def __init__(self, input_channels, primary_dim, num_classes, output_dim, num_routing, num_compartments):
        super(MultiCompartmentCapsuleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(d_model, 256, kernel_size=9, stride=1)
        self.primary_capsules = MultiCompartmentCapsuleLayer(num_compartments, 8, -1, 256, primary_dim, kernel_size=9, stride=2)
        self.digit_capsules = MultiCompartmentCapsuleLayer(num_compartments, num_classes, 32 * 6 * 6, primary_dim, output_dim)
        self.decoder = nn.Sequential(
            nn.Linear(16 * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        return classes

#End of structurization







class SpikingMultiheadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SpikingMultiheadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Ensure the embedding size is divisible by the number of heads
        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        # Linear transformations for values, keys, and queries
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Final fully connected output
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        # Neuron and STDP instances
        self.neuron = MultiCompartmentLIFNeuron(num_compartments)
        self.stdp = STDPLearning()

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Reshape for multi-head attention
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Get spikes for values, keys, and queries
        values_spikes = self.neuron(self.values(values))
        keys_spikes = self.neuron(self.keys(keys))
        queries_spikes = self.neuron(self.queries(queries))

        # Apply STDP learning
        dw_values = self.stdp(values_spikes, values_spikes)
        dw_keys = self.stdp(keys_spikes, keys_spikes)
        dw_queries = self.stdp(queries_spikes, queries_spikes)

        # Update weights (naive way, there might be a better way to integrate STDP learning rule with PyTorch optimizers)
        self.values.weight += dw_values
        self.keys.weight += dw_keys
        self.queries.weight += dw_queries

        # Normal attention mechanism
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries_spikes, keys_spikes])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values_spikes]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)

        return out

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


class SpikingAttentionWeight(nn.Module):
    def __init__(self, embed_size):
        super(SpikingAttentionWeight, self).__init__()
        self.scale_factor = torch.nn.Parameter(torch.sqrt(torch.tensor(embed_size, dtype=torch.float32)))

    def forward(self, energy, mask):
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.nn.functional.softmax(energy / self.scale_factor, dim=3)
        return attention

class SpikingWeightedSum(nn.Module):
    def __init__(self):
        super(SpikingWeightedSum, self).__init__()

    def forward(self, attention, values_spikes):
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values_spikes])
        return out

# Define the spiking multi-head attention mechanism
class SpikingMultiheadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SpikingMultiheadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.neuron = MultiCompartmentLIFNeuron(num_compartments)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Reshape for multi-head attention
        values = values.reshape(N, value_len, self.heads, -1)
        keys = keys.reshape(N, key_len, self.heads, -1)
        queries = query.reshape(N, query_len, self.heads, -1)

        # Get spikes for values, keys, and queries
        values_spikes = self.neuron(self.values(values))
        keys_spikes = self.neuron(self.keys(keys))
        queries_spikes = self.neuron(self.queries(queries))

        # Compute spiking dot product
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries_spikes, keys_spikes])

        # Compute spiking attention weights
        attention_weight_model = SpikingAttentionWeight(self.embed_size)
        attention = attention_weight_model(energy, mask)

        # Compute spiking weighted sum
        weighted_sum_model = SpikingWeightedSum()
        out_spikes = weighted_sum_model(attention, values_spikes)

        out = self.fc_out(out_spikes.reshape(N, query_len, self.heads * self.head_dim))
        return out

class SpikingNeuronLayer(nn.Module):
    def __init__(self, size):
        super(SpikingNeuronLayer, self).__init__()
        self.size = size
        self.membrane_potentials = None
        self.spike_trains = None

    def forward(self, x):
        if self.membrane_potentials is None:
            self.membrane_potentials = torch.zeros(*x.shape[:-1], self.size).to(x.device)
        if self.spike_trains is None:
            self.spike_trains = torch.zeros(*x.shape[:-1], self.size).to(x.device)
        self.membrane_potentials += x
        spikes = (self.membrane_potentials > SPIKE_THRESHOLD).float()
        self.spike_trains = spikes
        self.membrane_potentials = self.membrane_potentials * (1.0 - spikes)
        return spikes

class SpikingFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpikingFeedForward, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.neuron1 = SpikingNeuronLayer(hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.neuron2 = SpikingNeuronLayer(output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.neuron1(x)
        x = self.linear2(x)
        return self.neuron2(x)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = self._get_positional_encoding(d_model, max_len)

    def _get_positional_encoding(self, d_model, max_len):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos = position * div_term
        pos_embedding = torch.zeros(max_len, d_model)
        pos_embedding[:, 0::2] = torch.sin(pos)
        pos_embedding[:, 1::2] = torch.cos(pos)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def forward(self, token_embedding):
        # Ensure that the token_embedding shape is (batch_size, sequence_length, d_model)
        assert len(token_embedding.shape) == 3, "Expected the token_embedding shape to be (batch_size, sequence_length, d_model)"
        batch_size, seq_len, d_model = token_embedding.shape

        # Ensure the positional embedding is broadcasted along the batch_size dimension
        return token_embedding + self.pos_embedding[:, :d_model, :seq_len].permute(0, 2, 1)



def update_weights_snn(model, learning_rate):
    for module in model.modules():
        if isinstance(module, STDPLearning):
            delta_w = module.forward(module.pre_trace, module.post_trace)
            with torch.no_grad():
                module.weight += learning_rate * delta_w
                module.weight.clamp_(WEIGHT_MIN, WEIGHT_MAX)



# Transformer Block with MultiCompartmentLIFNeuron & STDP
class STDPTransformerBlock(nn.Module):
    def __init__(self, k, heads, head_dim=64):
        super(STDPTransformerBlock, self).__init__()

        # エンベッディング次元を増やすための全結合層
        self.expand_dim = nn.Linear(k, heads * head_dim)

        self.attention = SpikingMultiheadAttention(heads * head_dim, heads=heads)
        self.norm1 = nn.LayerNorm(heads * head_dim)
        self.norm2 = nn.LayerNorm(heads * head_dim)

        self.spike_layer1 = MultiCompartmentLIFNeuron(num_compartments)
        self.stdp1 = STDPLearning()

        self.ff = nn.Sequential(
            nn.Linear(heads * head_dim, heads * head_dim),
            nn.ReLU(),
            nn.Linear(heads * head_dim, heads * head_dim)
        )

        self.spike_layer2 = MultiCompartmentLIFNeuron(num_compartments)
        self.stdp2 = STDPLearning()

    def forward(self, x):
        x = x.reshape(x.size(0) * x.size(1), -1)  # 形状を[32*512, 10]に変更
        x = self.expand_dim(x)
        x = x.reshape(x.size(0) // 10, 512, 10)  # 形状を[32, 512, 10]に戻す

        attended = self.attention(x, x, x, None)
        x = self.norm1(attended + x)

        # Spike layer after attention
        spike_out1 = self.spike_layer1(x)
        dw1 = self.stdp1(x, spike_out1)
        x += dw1

        x = self.ff(x)
        x = self.norm2(x)

        # Spike layer after FFN
        spike_out2 = self.spike_layer2(x)
        dw2 = self.stdp2(x, spike_out2)
        x += dw2

        return x

# Encoder with MultiCompartmentLIFNeuron & STDP
class STDPEncoder(nn.Module):
    def __init__(self, k, heads, num_layers):
        super(STDPEncoder, self).__init__()
        self.num_layers = num_layers
        self.enc_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.enc_layers.append(STDPTransformerBlock(k, heads))
        self.pos_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        x = self.pos_encoding(x)
        x = x.transpose(1, 2)  # 形状を[32, 10, 512]に変更
        for layer in self.enc_layers:
            x = layer(x)
        return x

# Decoder with MultiCompartmentLIFNeuron & STDP
class STDPDecoder(nn.Module):
    def __init__(self, k, heads, num_layers):
        super(STDPDecoder, self).__init__()
        self.num_layers = num_layers
        self.dec_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.dec_layers.append(STDPTransformerBlock(k, heads))
        self.pos_encoding = PositionalEncoding(d_model)

    def forward(self, x, encoder_output):
        x = self.pos_encoding(x)
        for layer in self.dec_layers:
            x = layer(x + encoder_output)
        return x

# Transformer with MultiCompartmentLIFNeuron & STDP
class STDPTransformer(nn.Module):
    def __init__(self, k, heads, num_classes, num_layers=NUM_LAYERS):
        super().__init__()
        self.encoder = STDPEncoder(k, heads, num_layers)
        self.decoder = STDPDecoder(k, heads, num_layers)
        self.pos_enc = PositionalEncoding(k)
        self.fc = nn.Linear(k, num_classes)

    def forward(self, x, target):
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.decoder(x, target)
        return self.fc(x)
