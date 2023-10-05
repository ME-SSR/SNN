import torch
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.analysis.visualization import summary

# Simulation time.
time = 1000

# Create the network.
network = UShapedPipeTopology(dt = 1.0 , batch_size = 10, num_neurons = 5, shape = ([15,15]), radius = 2, u_depth = 0.1 )
u_shaped_network = network.build()  # これでネットワークのインスタンスを取得します

visualize_network(u_shaped_network)
print(summary(u_shaped_network))

# Create and add input and output layer monitors.
source_monitor = Monitor(
    obj=u_shaped_network.Neuron_0,
    state_vars=("s",),  # Record spikes and voltages.
    time=time,  # Length of simulation (if known ahead of time).
)
target_monitor = Monitor(
    obj=u_shaped_network.Neuron_4,
    state_vars=("s", "v"),  # Record spikes and voltages.
    time=time,  # Length of simulation (if known ahead of time).
)

u_shaped_network.add_monitor(monitor=source_monitor, name="Neuron_0")
u_shaped_network.add_monitor(monitor=target_monitor, name="Neuron_39")

# Create input spike data, where each spike is distributed according to Bernoulli(0.1).
input_data = torch.bernoulli(0.1 * torch.ones(time, u_shaped_network.Neuron_0.n)).byte()
inputs = {"Neuron_0": input_data}

# Simulate network on input data.
u_shaped_network.run(inputs=inputs, time=time)

# Retrieve and plot simulation spike, voltage data from monitors.
spikes = {
    "Neuron_0": source_monitor.get("s"), "Neuron_39": target_monitor.get("s")
}
voltages = {"Neuron_39": target_monitor.get("v")}

plt.ioff()
plot_spikes(spikes)
plot_voltages(voltages, plot_type="line")
plt.show()
