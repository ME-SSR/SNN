#データローダー、エンコーダー及びデコーダー、そしてモデルの構築
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random


# データセット定義
class RandomDataset(Dataset):
    def __init__(self, data_size, sequence_length, vocab_size):
        self.data_size = data_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return (
            torch.randint(0, self.vocab_size, (self.sequence_length,)),
            torch.randint(0, self.vocab_size, (self.sequence_length,))
        )

# データセットとデータローダーの初期化
dataset = RandomDataset(1000, 10, 512)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# モデル定義
d_model = 512

# 必要な変数の定義
k = d_model  # embed size
heads = 8  # attention heads
vocab_size = 512

# モデルの初期化
class STDPTransformerWithEmbeddings(nn.Module):
    def __init__(self, k, heads, num_classes, num_layers=NUM_LAYERS, vocab_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, k)  # Embedding layer
        self.encoder = STDPEncoder(k, heads, num_layers)
        self.capsule_network = MultiCompartmentCapsuleNetwork(input_channels=k, primary_dim=8, num_classes=num_classes, output_dim=16, num_routing=3, num_compartments=2)
        self.decoder = STDPDecoder(k, heads, num_layers)
        self.pos_enc = PositionalEncoding(k)
        self.fc = nn.Linear(k, num_classes)

    def forward(self, x, target):
        x = self.embedding(x)  # Use embedding layer
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, d_model, sequence_length)
        x = self.encoder(x)
        x = self.capsule_network(x)
        x = self.decoder(x, target)
        return self.fc(x)

model = STDPTransformerWithEmbeddings(k, heads, vocab_size, num_layers=NUM_LAYERS)

# サンプルデータをモデルに通して動作確認
sample_input, sample_output = next(iter(dataloader))
model_output = model(sample_input, sample_output)
print(model_output.shape)
