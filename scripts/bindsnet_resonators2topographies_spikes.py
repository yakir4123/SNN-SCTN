import math
import os
import pickle
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

try:
    from bindsnet.network import Network
except:
    from bindsnet.network import Network

from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import LIFNodes, Input
from bindsnet.network.topology import Connection


def batch_dicts(dicts):
    if len(dicts) == 0:
        raise ValueError("Input list is empty")

    # Get the keys from the first dictionary in the list
    keys = list(dicts[0].keys())

    # Initialize a result dictionary with the same keys
    result = {key: [] for key in keys}

    # Stack the values for each key
    for key in keys:
        values = [d[key] for d in dicts]
        stacked_values = torch.stack(values, dim=1).squeeze(dim=2)  # Stack along the new dimension (B)
        result[key] = stacked_values

    return result


class Resonators2BandsLayer(Network):

  def __init__(self, channel_name, band_name, clk_freq_resonators: torch.Tensor):
    super().__init__(dt=1)
    self.n_resonators = len(clk_freq_resonators)
    self.band_name = band_name
    self.channel_name = channel_name
    # Layers
    input_layer = Input(
        n=self.n_resonators,
        shape=(self.n_resonators, ),
        traces=True
        , tc_trace=20.0
    )

    agg_layer = LIFNodes(
        n=1,
        traces=False,
        rest=0.0,
        reset=3 * self.n_resonators,
        thresh=5 * self.n_resonators,
        tc_decay=60 * self.n_resonators,
        refrac=1,
    )
    w = torch.where(clk_freq_resonators == 153600, 1. , 11.)
    input_exc_conn = Connection(
            source=input_layer,
            target=agg_layer,
            w=w,
        )
    self.add_layer(input_layer, name=f"{channel_name}-{band_name}-resonators")
    self.add_layer(agg_layer, name=f"{channel_name}-{band_name}-band")

    self.add_connection(input_exc_conn, source=f"{channel_name}-{band_name}-resonators", target=f"{channel_name}-{band_name}-band")

class Bands2TopographyLayer(Network):

  def __init__(self, band_name: str, ch_netowrks: dict, shape: Tuple, ch_pos: dict, sigma=1):
    super().__init__(dt=1)

    self.ch_netowrks = ch_netowrks
    self.band_name = band_name
    topography_layer = LIFNodes(
        shape=shape,
        n=math.prod(shape),
        traces=True,
        rest=0.0,
        thresh=4,
    )

    # add layers ch_network to topography network!
    for ch in ch_netowrks.keys():
      ch_network = ch_netowrks[ch]
      self.add_layer(ch_network.layers[f'{ch}-{band_name}-resonators'],
                     name=f'{ch}-{band_name}-resonators')
    for ch in ch_netowrks.keys():
      ch_network = ch_netowrks[ch]
      self.add_layer(ch_network.layers[f'{ch}-{band_name}-band'],
                     name=f'{ch}-{band_name}-band')

    self.add_layer(topography_layer, name=f"{band_name}-topography")
    for ch in ch_netowrks.keys():
      ch_network = ch_netowrks[ch]
      for (source, target), conn in ch_network.connections.items():
        self.add_connection(conn, source=source, target=target)

      output_ch_layer = ch_network.layers[f'{ch}-{band_name}-band']
      w = self.channel_weights(shape, center=ch_pos[ch], sigma=sigma)
      conn = Connection(
            source=output_ch_layer,
            target=topography_layer,
            w=w,
        )
      self.add_connection(conn, source=f'{ch}-{band_name}-band', target=f"{band_name}-topography")


  def channel_weights(self, kernel_shape, center, sigma=1):
    # Create an empty kernel
    weights = torch.zeros(kernel_shape)

    # Calculate the Gaussian values for the new kernel
    radius = (kernel_shape[0]-1)/2
    dr = 1
    xy_center = ((kernel_shape[0]-1)/2, (kernel_shape[1]-1)/2)
    for x in range(kernel_shape[0]):
        for y in range(kernel_shape[1]):
            distance_squared = (x - center[0])**2 + (y - center[1])**2
            weights[x, y] = np.exp(-distance_squared / (2 * sigma**2))

            r = np.sqrt((x - xy_center[0])**2 + (y - xy_center[1])**2)
            if (r - dr/2) > radius:
                weights[x, y] = 0

    # Normalize the kernel
    weights /= weights.max()
    # return weights
    return weights.view(1, -1)


# sim_time = 100
sim_time = 153600//4
full_resonator_array = torch.Tensor([
    1.1, 1.3, 1.6, 1.9, 2.2, 2.5,
    2.88, 3.05, 3.39, 3.7, 4.12, 4.62,
    5.09, 5.45, 5.87, 6.36, 6.8, 7.6,
    8.6, 10.5, 11.5, 12.8, 15.8, 16.6,
    19.4, 22.0, 24.8, 28.4, 30.5, 34.7,
    37.2, 40.2, 43.2, 47.7, 52.6, 57.2
    ])
bands = {
    'Delta': (.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 14),
    'Beta': (14, 32),
    'Gamma': (32, 62),
}


N = 11
xs = (np.array([-.7, -.66, 0, 0.2, 0.4, 0.5, 0.7, 0.7, 0.5, 0.4, 0.2, 0, -.66 ,-.7])+1)/2
ys = (-np.array([0.2, 0.6, .95, 0.6, 0.3, 0.6, 0.2, -0.2, -0.6, -0.3, -0.6, -.95, -0.6, -0.2])+1)/2
channels = ['O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', ]

ch_pos = {
    ch: (x, y)
    for ch, (x, y) in zip(channels, zip(ys, xs))
}

ch_pos_N = {k: (int(x * N), int(y * N)) for k, (x, y) in ch_pos.items()}
map_pos2ch = {(vx, vy): k for k, (vx, vy) in ch_pos_N.items()}

def create_topography_network(band_name):
    (lf, hf) = bands[band_name]
    resonators2band = {}
    for ch in ch_pos.keys():
        freqs = full_resonator_array[(full_resonator_array >= lf) &
                                     (full_resonator_array< hf)]
        clk_freqs = torch.where(freqs < 10, 15360, 153600)
        resonators2band[ch] = Resonators2BandsLayer(ch, band_name, clk_freqs)

    topography_network = Bands2TopographyLayer(band_name, resonators2band, shape=(N, N), ch_pos=ch_pos_N)
    target_monitor = Monitor(
            obj=topography_network.layers[f"{band_name}-topography"],
            state_vars=("s",),
            time=sim_time,
        )
    topography_network.add_monitor(monitor=target_monitor, name=f"{band_name}-topography")
    return topography_network

# Run only on this band.
main_path = Path('../datasets/EEG_data_for_Mental_Attention_State_Detection/train_test_dataset')
output_path_dir = Path('../datasets/EEG_data_for_Mental_Attention_State_Detection/preprocessed_resonators')

trial = '3'
minute = 27
if minute < 10:
    label = 'focus'
elif minute < 20:
    label = 'unfocus'
else:
    label = 'drowsed'

band_name = 'Gamma'
data = []
for fname in os.listdir(main_path / trial / label):
    milli_second = float(fname[:-4])/60
    if not (minute <= milli_second < (minute+1)):
        continue
    with open(main_path / trial / label / fname, 'rb') as f:
        data.append({})
        for k, t in pickle.load(f).items():
            if k.endswith(band_name):
                loaded_data = t.T.reshape(t.shape[1], 1, t.shape[0])[:sim_time]
                data[-1][f'{k}-resonators'] = loaded_data
batch = batch_dicts(data)
snn = create_topography_network(band_name)

label_tensor = torch.ones(list(batch.values())[0].shape[1]) * (minute//10)
t1 = time.time()
snn.run(inputs=batch, time=sim_time)
t2 = time.time()
print(t2 - t1)
output_path_file = output_path_dir / trial / band_name / f'{minute}.pt'
output_path_file.parent.mkdir(parents=True, exist_ok=True)

print(output_path_file)
torch.save(list(snn.monitors.values())[0].get("s"), output_path_file)
