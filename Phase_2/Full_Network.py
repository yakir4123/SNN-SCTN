import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from numba.typed import List as numbaList
from Phase_2.poission_encoding import process_accel_gyro_data
from Phase_2.utils_phase2 import breakdown_data, get_gt_data, \
    generate_and_plot_parameter_arrays
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import create_SCTN, SCTNeuron
from snn.graphs import plot_network
from snn.layers import SCTNLayer

plt.switch_backend('Qt5Agg')

# Constants
GRID_SIZE = 50
THETA = -0.45
LP_RANGE = (50, 2)
LF_CONST = 1



def create_flattened_neuron_grid(weights_X, weights_Y, LP_array, LF, Theta, threshold_pulse=2):
    flattened_neurons = numbaList()
    for i in range(weights_X.shape[0]):
        for j in range(weights_X.shape[1]):
            neuron = SCTNeuron(
                synapses_weights=np.array([weights_X[i, j], weights_Y[i, j], 0, 0]),
                leakage_factor=int(LF),
                leakage_period=int(LP_array[i, j]),
                theta=Theta,
                activation_function=1,  # Using BINARY activation
                threshold_pulse=threshold_pulse,
                log_membrane_potential=True,
                log_out_spikes=1
            )
            flattened_neurons.append(neuron)

    for i in range(weights_X.shape[0]):
        for j in range(weights_X.shape[1]):
            neuron = SCTNeuron(
                synapses_weights=np.array([0, 0, weights_X[i, j], weights_Y[i, j]]),
                leakage_factor=int(LF),
                leakage_period=int(LP_array[i, j]),
                theta=Theta,
                activation_function=1,  # Using BINARY activation
                threshold_pulse=threshold_pulse,
                log_membrane_potential=True,
                log_out_spikes=1
            )
            flattened_neurons.append(neuron)
    return flattened_neurons


def create_next_layer(weights, LP, LF, Theta, threshold_pulse=2):
    flattened_neurons = numbaList()
    for i in range(weights.shape[0]):
        neuron = SCTNeuron(
            synapses_weights=weights[i],  # Use the entire row of weights for each neuron
            leakage_factor=int(LF),
            leakage_period=int(LP),
            theta=Theta,
            activation_function=1,  # Using BINARY activation
            threshold_pulse=threshold_pulse,
            log_membrane_potential=True
        )
        flattened_neurons.append(neuron)
    return flattened_neurons


def create_next_level_weight_matrix(grid_size=50, next_layer_size=9, plot_weights=0,w_amplifier=1):
    full_length = grid_size * grid_size * 2
    half_length = grid_size * grid_size
    quarter_length = grid_size * grid_size // 2
    three_quarters = half_length + quarter_length

    sequence_1_lead = np.tile([1] * (grid_size // 2) + [0] * (grid_size // 2), quarter_length // (grid_size))
    sequence_1_lead_full = np.tile([1] * (grid_size // 2) + [0] * (grid_size // 2), half_length // (grid_size))
    sequence_0_lead = np.tile([0] * (grid_size // 2) + [1] * (grid_size // 2), quarter_length // (grid_size))
    sequence_0_lead_full = np.tile([0] * (grid_size // 2) + [1] * (grid_size // 2), half_length // (grid_size))

    weights = np.zeros((next_layer_size, full_length))

    weights[0, half_length:three_quarters] = sequence_1_lead * w_amplifier
    weights[1, 0:quarter_length] = 0.5 * w_amplifier
    weights[2, half_length:three_quarters] = sequence_0_lead * w_amplifier
    weights[3, 0:half_length] = sequence_1_lead_full * 0.5 * w_amplifier
    weights[4, :] = 0
    weights[5, :half_length] = sequence_0_lead_full * 0.5 * w_amplifier
    weights[6, three_quarters:] = sequence_1_lead * w_amplifier
    weights[7, quarter_length:half_length] = 0.5 * w_amplifier
    weights[8, three_quarters:] = sequence_0_lead * w_amplifier

    if plot_weights:
        plot_neuron_weights(weights, grid_size)
    return weights


def plot_neuron_weights(weights, grid_size=50):
    fig, axs = plt.subplots(3, 3, figsize=(16, 16))  # 20% smaller than 20x20
    fig.suptitle('Input Weights for 9 Neurons', fontsize=12.8)  # 20% smaller than 16

    # Custom colormaps
    cmap_accel = LinearSegmentedColormap.from_list("custom_accel", ["white", "red"])
    cmap_gyro = LinearSegmentedColormap.from_list("custom_gyro", ["white", "green"])

    for i in range(9):
        row = i // 3
        col = i % 3

        if i + 1 in [2, 4, 6, 8]:  # Acceleration only
            weights_to_plot = weights[i, :grid_size * grid_size].reshape(grid_size, grid_size)
            title = f'Neuron {i + 1} - Acceleration'
            cmap = cmap_accel
        elif i + 1 in [1, 3, 7, 9]:  # Gyroscope only
            weights_to_plot = weights[i, grid_size * grid_size:].reshape(grid_size, grid_size)
            title = f'Neuron {i + 1} - Gyroscope'
            cmap = cmap_gyro
        else:  # Neuron 5, show both
            weights_accel = weights[i, :grid_size * grid_size].reshape(grid_size, grid_size)
            weights_gyro = weights[i, grid_size * grid_size:].reshape(grid_size, grid_size)
            weights_to_plot = np.hstack((weights_accel, weights_gyro))
            title = f'Neuron {i + 1} - Accel | Gyro'
            cmap = 'viridis'

        im = axs[row, col].imshow(weights_to_plot, cmap=cmap, aspect='equal')
        axs[row, col].set_title(title, fontsize=9.6)  # 20% smaller than 12

        # Add x and y axis scales
        axs[row, col].set_xticks(np.arange(0, weights_to_plot.shape[1], 10))
        axs[row, col].set_yticks(np.arange(0, weights_to_plot.shape[0], 10))
        axs[row, col].set_xticklabels(np.arange(0, weights_to_plot.shape[1], 10), fontsize=8)
        axs[row, col].set_yticklabels(np.arange(0, weights_to_plot.shape[0], 10), fontsize=8)

        # Add colorbar
        cbar = plt.colorbar(im, ax=axs[row, col], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()


def get_out_spikes_array(network, processed_data, grid_size=50):
    start_iteration = 2 * grid_size * grid_size
    end_iteration = start_iteration + 8
    num_samples = processed_data.shape[0]
    print('Number of samples are: ', num_samples)
    spikes_array = np.zeros((9, num_samples))

    # Create a tqdm progress bar
    pbar = tqdm(enumerate(range(start_iteration, end_iteration + 1)), total=9, desc="Processing neurons")

    for i, neuron_index in pbar:
        # Update the progress bar description
        pbar.set_description(f"Processing neuron {i+1}/9")

        # Assuming out_spikes returns an array of spikes
        neuron_spikes = network.neurons[neuron_index].out_spikes()
        # Pad or truncate the neuron_spikes to match num_samples
        if len(neuron_spikes) > num_samples:
            spikes_array[i, :] = neuron_spikes[:num_samples]
        else:
            spikes_array[i, :len(neuron_spikes)] = neuron_spikes

    return spikes_array


def plot_neuron_spike_activity(network, neuron_index=1350, time_window=0.01, start_time=0, end_time=None):
    """
    Plot the spike activity of a specific neuron in the network, showing each sample.

    Parameters:
    network: The neural network object
    neuron_index: Index of the neuron to visualize (default is 1350)
    time_window: Time step between samples in seconds (default is 0.01s)
    start_time: Start time for the plot in seconds (default is 0)
    end_time: End time for the plot in seconds (default is None, which plots all data)
    """
    # Extract spike data for the specified neuron
    neuron_spikes = network.neurons[neuron_index].out_spikes()
    total_samples = len(neuron_spikes)
    total_time = total_samples * time_window

    print(f"Total dataset duration: {total_time:.2f} seconds")
    print(f"Total number of samples: {total_samples}")

    # Convert start_time and end_time to sample indices
    start_sample = int(start_time / time_window)
    end_sample = int(end_time / time_window) if end_time is not None else total_samples

    # Ensure end_sample doesn't exceed the data length
    end_sample = min(end_sample, total_samples)

    # Create time array for the selected range
    time = np.arange(start_sample, end_sample) * time_window

    # Create the plot
    plt.figure(figsize=(20, 6))
    plt.plot(time, neuron_spikes[start_sample:end_sample], 'b-', linewidth=2, drawstyle='steps-post')
    plt.title(f'Spike Activity of Neuron {neuron_index}')
    plt.xlabel('Time (s)')
    plt.ylabel('Spike (0 or 1)')
    plt.ylim(-0.1, 1.1)  # Set y-axis limits for binary spikes
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add a text box with spike statistics
    plotted_spikes = neuron_spikes[start_sample:end_sample]
    total_spikes = np.sum(plotted_spikes)
    spike_rate = total_spikes / ((end_sample - start_sample) * time_window)
    stats_text = (f'Plotted Time Range: {time[0]:.2f}s - {time[-1]:.2f}s\n'
                  f'Samples Plotted: {len(plotted_spikes)}\n'
                  f'Total Spikes: {total_spikes}\n'
                  f'Spike Rate: {spike_rate:.2f} Hz')
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def log_network_spikes(network, GRID_SIZE):
    start_iteration = 2 * GRID_SIZE * GRID_SIZE
    end_iteration = start_iteration + 8

    for i in range(start_iteration, end_iteration + 1):
        print(f"Logging spikes for iteration {i}")
        network.log_membrane_potential(i)
        network.log_out_spikes(i)


def plot_neuron_output_spikes(spikes_array):
    """
    Plot the output spikes of each neuron in separate subplots within a single window.

    Parameters:
    spikes_array (numpy.ndarray): A 2D array where each row represents a neuron's output spikes over time.
    """
    num_neurons, num_samples = spikes_array.shape

    # Calculate the number of rows and columns for the subplots
    num_rows = int(np.ceil(np.sqrt(num_neurons)))
    num_cols = int(np.ceil(num_neurons / num_rows))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    fig.suptitle('Output Spikes for Each Neuron', fontsize=16)

    for i in range(num_neurons):
        row = i // num_cols
        col = i % num_cols

        if num_rows == 1:
            ax = axs[col]
        elif num_cols == 1:
            ax = axs[row]
        else:
            ax = axs[row, col]

        ax.plot(spikes_array[i], drawstyle='steps-post')
        ax.set_title(f'Neuron {i + 1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Spike')
        ax.set_ylim(-0.1, 1.1)  # Set y-axis limits for binary spikes
        ax.grid(True)

    # Remove any unused subplots
    for i in range(num_neurons, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        if num_rows == 1:
            fig.delaxes(axs[col])
        elif num_cols == 1:
            fig.delaxes(axs[row])
        else:
            fig.delaxes(axs[row, col])

    plt.tight_layout()
    plt.show()



# Load and process data (Phase 1)
dir_path = r'C:\Users\user1\PycharmProjects\SNN-SCTN\Phase_2\Uzh_datasets\indoor_forward_9_davis'
ts, gyro, accel = breakdown_data(dir_path, start=27264,
                                 stop=56032,
                                 plot_data=0,
                                 fix_axis_and_g=1,
                                 fix_meas=0,
                                 no_Z_accel=0,
                                 no_Z_gyro=0)
gt_ts, gt_t_xyz, gt_q_wxyz, gt_data, all_translated_points = get_gt_data(dir_path,plot_data=1)
network_input = process_accel_gyro_data(accel=accel,
                                        gyro=gyro,
                                        plot_data=1)
print('Loaded data. Input size is: ', network_input.shape)


# Create full 2 layer network- (Phases 2 & 3)
weights_X, weights_Y, LP_array, LF, Theta = generate_and_plot_parameter_arrays(
                                            plot_param=1,
                                            grid_size=GRID_SIZE,
                                            theta=THETA,
                                            lp_range=LP_RANGE,
                                            lf=LF_CONST,
                                            w_amplifier=3)

fullNetwork = SpikingNetwork()

neuron_layer_accel_and_gyro = create_flattened_neuron_grid(weights_X=weights_X,
                                                           weights_Y=weights_Y,
                                                           LP_array=LP_array,
                                                           LF=LF,
                                                           Theta=Theta,
                                                           threshold_pulse=2)
fullNetwork.add_layer(SCTNLayer(neuron_layer_accel_and_gyro))

weights_for_next_layer = create_next_level_weight_matrix(grid_size=GRID_SIZE,
                                                         plot_weights=1,
                                                         w_amplifier=5)
distance_layer = create_next_layer(weights=weights_for_next_layer,
                                   LP=0,
                                   LF=0,
                                   Theta=0)

fullNetwork.add_layer(SCTNLayer(distance_layer))
print('Created a two-layer SpikingNetwork')

# Insert data

fullNetwork.input_full_data_spikes(network_input)

# plot_first_layer_spikes(fullNetwork, GRID_SIZE)
plot_neuron_spike_activity(fullNetwork, neuron_index=1350, time_window=0.001)

output_spikes_layer2 = get_out_spikes_array(fullNetwork,network_input, GRID_SIZE)

print('Inserted full dataset to network')
# # # Won't work for big grids (over
# plot_network(fullNetwork)
# plt.title('Plot network')
# plt.show()




print('Ploting second layer neurons output..')
plot_neuron_output_spikes(output_spikes_layer2)

