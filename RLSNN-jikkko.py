#データローダー、エンコーダー及びデコーダー、そしてモデルの構築
import torch
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.analysis.pipeline_analysis import MatplotlibAnalyzer
#from bindsnet.analysis.visualization import summary


torch_fix_seed()

# Simulation time.
time = 1000
spike_rate = 0.5

text = "hogehogehogeほげほげほげー"
max_size = 780
hidden_dim = 4
bottleneck_dim = 1
InputShape = [1,4]

autoencoder = Autoencoder(max_size, hidden_dim, bottleneck_dim)
input_data = string_to_normalized_tensor(text, max_size)
encoded_data = autoencoder.encoder(input_data)

snn_encoder = SNNEncoder(time, spike_rate)
spike_data = snn_encoder.encode(encoded_data)

# Create the network.
network = UShapedPipeTopology(dt = 1.0 , batch_size = 1, num_neurons = 6, shape = ([hidden_dim,bottleneck_dim]), radius = 2, u_depth = 0.1, net_thickness = 3 , net_interval = 2 )
u_shaped_network = network.build()  # これでネットワークのインスタンスを取得します

visualize_network(u_shaped_network)
#print(summary(u_shaped_network))

# Create and add input and output layer monitors.
source_monitor = Monitor(
    obj=u_shaped_network.Input,
    state_vars=("s"),  # Record spikes and voltages.
    time=time,  # Length of simulation (if known ahead of time).
)
target_monitor = Monitor(
    obj=u_shaped_network.Neuron_0,
    state_vars=("s", "v"),  # Record spikes and voltages.
    time=time,  # Length of simulation (if known ahead of time).
)

u_shaped_network.add_monitor(monitor=source_monitor, name="Input")
u_shaped_network.add_monitor(monitor=target_monitor, name="Output")

# Create input spike data, where each spike is distributed according to Bernoulli(0.1).
spike_data = torch.bernoulli(0.1 * torch.ones(time, u_shaped_network.Input.n)).byte()
inputs = {"Input": spike_data}

# Simulate network on input data.
u_shaped_network.run(inputs=inputs, time=time)
snn_output = target_monitor.get("s")

# 4. SNNの出力をSNNデコーダーでデコード
snn_decoder = SNNDecoder()
decoded_snn_output = snn_decoder.decode(snn_output)

# 5. オートエンコーダーのデコーダー部分で元のデータを再構築
reconstructed_data = autoencoder.decoder(decoded_snn_output)
output = tensor_to_strings(reconstructed_data)

print(output)

# Retrieve and plot simulation spike, voltage data from monitors.
spikes = {
    "Input": source_monitor.get("s"), "Output": target_monitor.get("s")
}
voltages = {"Output": target_monitor.get("v")}

plt.ioff()
plot_spikes(spikes)
plot_voltages(voltages, plot_type="line")
plt.show()
