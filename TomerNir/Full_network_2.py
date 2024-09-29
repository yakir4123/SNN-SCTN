import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from numba.typed import List as numbaList

from Phase_2.gridV3 import create_neuron_grid
from Phase_2.poission_encoding import process_accel_gyro_data
from Phase_2.utils_phase2 import breakdown_data, get_gt_data, \
    generate_and_plot_parameter_arrays
from snn.spiking_network import SpikingNetwork
from snn.spiking_neuron import create_SCTN, SCTNeuron
from snn.graphs import plot_network
from snn.layers import SCTNLayer
import os
from datetime import datetime

plt.switch_backend('Qt5Agg')

# Constants
GRID_SIZE = 50
THETA = -0.45
LP_RANGE = (50, 2)
LF_CONST = 1
LF_layer_2 = 0.7
LP_LAYER_2 = 50
THETA_LAYER_2 = 0
THRESHOLD = 1


def calculate_grid_spikes(neuron_grid, input_data):
    """
    Calculate spikes for each neuron in the grid for all input samples.

    Parameters:
    neuron_grid (numpy.ndarray): 2D array of neurons with shape (GRID_SIZE, GRID_SIZE)
    input_data (numpy.ndarray): Input data with shape (num_samples, 2)

    Returns:
    numpy.ndarray: Flattened array of spikes with shape (GRID_SIZE*GRID_SIZE, num_samples)
    """
    GRID_SIZE = neuron_grid.shape[0]
    num_samples = input_data.shape[0]

    # Initialize the output array
    spike_array = np.zeros((GRID_SIZE * GRID_SIZE, num_samples))

    # Iterate through all samples with a progress bar
    for sample_idx in tqdm(range(num_samples), desc="Processing samples"):
        x, y = input_data[sample_idx]
        input_sample = np.array([x, y])

        # Iterate through all neurons in the grid
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                neuron = neuron_grid[i, j]
                spike = neuron.ctn_cycle(input_sample, True)
                spike_array[i * GRID_SIZE + j, sample_idx] = 1 if spike else 0

    return spike_array


def combine_accel_gyro_spikes(accel_spikes, gyro_spikes):
    """
    Combine spike arrays from accelerometer and gyroscope grids into a single array.

    Parameters:
    accel_spikes (numpy.ndarray): Spike array from accelerometer grid with shape (GRID_SIZE*GRID_SIZE, num_samples)
    gyro_spikes (numpy.ndarray): Spike array from gyroscope grid with shape (GRID_SIZE*GRID_SIZE, num_samples)

    Returns:
    numpy.ndarray: Combined spike array with shape (2*GRID_SIZE*GRID_SIZE, num_samples)
    """
    # Ensure that both input arrays have the same number of samples
    assert accel_spikes.shape[1] == gyro_spikes.shape[1], "Mismatch in number of samples between accel and gyro data"

    # Combine the arrays vertically
    combined_spikes = np.vstack([accel_spikes, gyro_spikes])

    print(f"Combined spike array shape: {combined_spikes.shape}")

    return combined_spikes


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


# def create_neuron_grid(weights, LP_array, LF, Theta, threshold_pulse=0):
#     neuron_grid = np.empty((3, 3), dtype=object)
#     edge = GRID_SIZE // 2
#     for i in range(-edge, edge):
#         for j in range(-edge, edge):
#             neuron = SCTNeuron(
#                 synapses_weights=np.array([weights[i]]),
#                 leakage_factor=LF,
#                 leakage_period=int(LP_array[i + edge, j + edge]),
#                 theta=Theta,
#                 activation_function=1,  # Using BINARY activation
#                 threshold_pulse=threshold_pulse,
#                 log_membrane_potential=True
#             )
#             neuron_grid[i + edge, j + edge] = neuron
#     return neuron_grid


# Load and process data (Phase 1)


def create_next_layer(grid_size=50, next_layer_size=9,plot=0,w_amp_accel=1, w_amp_gyro=1,LF_layer_2=0,LP_LAYER_2=0,THETA_LAYER_2=0,threshold_pulse=0):
    """
    Create the next layer of neurons (3x3 grid).

    Parameters:
    grid_size (int): Size of the previous layer's grid
    next_layer_size (int): Size of the next layer (default is 9 for 3x3 grid)

    Returns:
    list: List of SCTNeuron objects for the next layer
    """
    full_length = grid_size * grid_size * 2
    half_length = grid_size * grid_size
    quarter_length = grid_size * grid_size // 2
    three_quarters = half_length + quarter_length

    sequence_1_lead = np.tile([1] * (grid_size // 2) + [0] * (grid_size // 2), quarter_length // (grid_size))
    sequence_1_lead_full = np.tile([1] * (grid_size // 2) + [0] * (grid_size // 2), half_length // (grid_size))
    sequence_0_lead = np.tile([0] * (grid_size // 2) + [1] * (grid_size // 2), quarter_length // (grid_size))
    sequence_0_lead_full = np.tile([0] * (grid_size // 2) + [1] * (grid_size // 2), half_length // (grid_size))

    weights = np.zeros((next_layer_size, full_length))

    weights[0, half_length:three_quarters] = sequence_1_lead * w_amp_gyro
    weights[1, 0:quarter_length] = w_amp_accel
    weights[2, half_length:three_quarters] = sequence_0_lead * w_amp_gyro
    weights[3, 0:half_length] = sequence_1_lead_full * w_amp_accel
    weights[4, :] = 0
    weights[5, :half_length] = sequence_0_lead_full * w_amp_accel
    weights[6, three_quarters:] = sequence_1_lead * w_amp_gyro
    weights[7, quarter_length:half_length] = w_amp_accel
    weights[8, three_quarters:] = sequence_0_lead * w_amp_gyro

    next_layer_neurons = numbaList()
    for i in range(next_layer_size):
        neuron = SCTNeuron(
            synapses_weights=weights[i],
            leakage_factor=LF_layer_2,
            leakage_period=LP_LAYER_2,
            theta=THETA_LAYER_2,
            activation_function=1,  # Using BINARY activation
            threshold_pulse=threshold_pulse,
            log_membrane_potential=True
        )
        next_layer_neurons.append(neuron)
    if plot:
        plot_neuron_weights(weights,grid_size)

    return next_layer_neurons


def process_next_layer(next_layer_neurons, combined_spikes):
    """
    Process the combined spikes through the next layer of neurons with a progress bar.

    Parameters:
    next_layer_neurons (list): List of SCTNeuron objects for the next layer
    combined_spikes (numpy.ndarray): Combined spike array from previous layer

    Returns:
    numpy.ndarray: Output spikes from the next layer
    """
    num_neurons = len(next_layer_neurons)
    num_samples = combined_spikes.shape[1]
    output_spikes = np.zeros((num_neurons, num_samples))

    # Create a tqdm progress bar
    with tqdm(total=num_samples, desc="Processing next layer") as pbar:
        for t in range(num_samples):
            input_spikes = combined_spikes[:, t]
            for i, neuron in enumerate(next_layer_neurons):
                spike = neuron.ctn_cycle(input_spikes, True)
                output_spikes[i, t] = 1 if spike else 0
            pbar.update(1)  # Update the progress bar

    return output_spikes


def plot_drone_movement(output_spikes, samples_window=10, step_size=0.1):
    """
    Plot the movement of the drone based on output spikes from the neural network.

    Parameters:
    output_spikes (numpy.ndarray): Array of output spikes with shape (9, num_samples)
    samples_window (int): Number of samples to consider for each movement step
    step_size (float): Size of each movement step
    """
    num_neurons, num_samples = output_spikes.shape
    num_steps = num_samples // samples_window

    # Define movement directions for each neuron
    directions = [
        (0, 1),  # North
        (1, 1),  # Northeast
        (1, 0),  # East
        (1, -1),  # Southeast
        (0, 0),  # No movement
        (-1, -1),  # Southwest
        (-1, 0),  # West
        (-1, 1),  # Northwest
        (0, -1)  # South
    ]

    # Initialize drone position
    position = np.zeros((num_steps + 1, 2))

    for step in range(num_steps):
        start_sample = step * samples_window
        end_sample = (step + 1) * samples_window

        # Count spikes in the current window for each neuron
        spike_counts = np.sum(output_spikes[:, start_sample:end_sample], axis=1)

        # Find the neuron with the most spikes
        dominant_neuron = np.argmax(spike_counts)

        # Update position based on the dominant neuron
        movement = np.array(directions[dominant_neuron]) * step_size
        position[step + 1] = position[step] + movement

    # Plot the movement
    plt.figure(figsize=(10, 10))
    plt.plot(position[:, 0], position[:, 1], 'b-')
    plt.plot(position[0, 0], position[0, 1], 'go', markersize=10, label='Start')
    plt.plot(position[-1, 0], position[-1, 1], 'ro', markersize=10, label='End')
    plt.title('Drone Movement')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    return position

def plot_output_neuron_spikes(output_spikes, time_window=0.01):
    """
    Plot the spikes of all 9 output neurons in one window.

    Parameters:
    output_spikes (numpy.ndarray): Array of output spikes with shape (9, num_samples)
    time_window (float): Time step between samples in seconds (default is 0.01s)
    """
    num_neurons, num_samples = output_spikes.shape
    time = np.arange(num_samples) * time_window

    plt.figure(figsize=(20, 12))
    for i in range(num_neurons):
        plt.subplot(9, 1, i + 1)
        plt.plot(time, output_spikes[i], drawstyle='steps-post', linewidth=1)
        plt.ylabel(f'Neuron {i + 1}')
        plt.ylim(-0.1, 1.1)
        if i < 8:
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        if i == 8:
            plt.xlabel('Time (s)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.title(f'Neuron {i + 1} Spike Activity', fontsize=10)

    plt.tight_layout()
    plt.suptitle('Output Spikes of All 9 Neurons', fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()


def save_output_spikes(output_spikes, LF_layer_2, LP_LAYER_2, THETA_LAYER_2, THRESHOLD):
    """
    Save the output spikes array to a file in the specified path with a unique filename.

    Parameters:
    output_spikes (numpy.ndarray): Array of output spikes to save
    LF_layer_2 (float): Constant value for LF_layer_2
    LP_LAYER_2 (float): Constant value for LP_LAYER_2
    THETA_LAYER_2 (float): Constant value for THETA_LAYER_2
    THRESHOLD (float): Constant value for THRESHOLD
    """
    # Define the base path
    base_path = r'C:\Users\user1\PycharmProjects\SNN-SCTN\Phase_3\output_spikes_arrays'

    # Create the directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)

    # Generate a unique filename based on constants and current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_spikes_LF{LF_layer_2}_LP{LP_LAYER_2}_THETA{THETA_LAYER_2}_THRESH{THRESHOLD}_{timestamp}.npy"

    # Full path for the file
    full_path = os.path.join(base_path, filename)

    # Save the array
    np.save(full_path, output_spikes)
    print(f"Output spikes saved to {full_path}")

def calculate_drone_trajectory(output_spikes, samples_window=10, linear_step=0.1, angular_step=np.pi / 18):
    """
    Calculate and plot the trajectory of the drone based on output spikes.

    Parameters:
    output_spikes (numpy.ndarray): Array of output spikes with shape (9, num_samples)
    samples_window (int): Number of samples to consider for each movement step
    linear_step (float): Size of each linear movement step
    angular_step (float): Size of each angular movement step (in radians)

    Returns:
    numpy.ndarray: Array of drone positions
    """
    num_neurons, num_samples = output_spikes.shape
    num_steps = num_samples // samples_window

    # Initialize drone state
    position = np.zeros((num_steps + 1, 2))
    orientation = 0  # in radians

    # Define neuron indices for angular and linear velocities
    angular_neurons = [0, 2, 6, 8]
    linear_neurons = [1, 3, 5, 7]

    for step in range(num_steps):
        start_sample = step * samples_window
        end_sample = (step + 1) * samples_window

        # Count spikes in the current window for each neuron
        spike_counts = np.sum(output_spikes[:, start_sample:end_sample], axis=1)

        # Calculate angular velocity
        angular_velocity = (spike_counts[0] - spike_counts[2] + spike_counts[6] - spike_counts[8]) * angular_step
        orientation += angular_velocity

        # Calculate linear velocity
        forward_velocity = (spike_counts[1] + spike_counts[3] - spike_counts[5] - spike_counts[7]) * linear_step

        # Update position based on orientation and forward velocity
        position[step + 1, 0] = position[step, 0] + forward_velocity * np.cos(orientation)
        position[step + 1, 1] = position[step, 1] + forward_velocity * np.sin(orientation)

    # Plot the trajectory
    plt.figure(figsize=(12, 10))
    plt.plot(position[:, 0], position[:, 1], 'b-')
    plt.plot(position[0, 0], position[0, 1], 'go', markersize=10, label='Start')
    plt.plot(position[-1, 0], position[-1, 1], 'ro', markersize=10, label='End')

    # Plot orientation arrows at regular intervals
    arrow_indices = np.linspace(0, num_steps, 20, dtype=int)
    for i in arrow_indices:
        dx = np.cos(orientation) * 0.5
        dy = np.sin(orientation) * 0.5
        plt.arrow(position[i, 0], position[i, 1], dx, dy, head_width=0.2, head_length=0.3, fc='r', ec='r')

    plt.title('Drone Trajectory')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    return position


dir_path = r'C:\Users\user1\PycharmProjects\SNN-SCTN\Phase_2\Uzh_datasets\indoor_forward_9_davis'
ts, gyro, accel = breakdown_data(dir_path, start=27264,
                                 stop=56032,
                                 plot_data=0,
                                 fix_axis_and_g=1,
                                 fix_meas=0,
                                 no_Z_accel=0,
                                 no_Z_gyro=0)
gt_ts, gt_t_xyz, gt_q_wxyz, gt_data, all_translated_points = get_gt_data(dir_path,plot_data=0)
accel_input, gyro_input = process_accel_gyro_data(accel, gyro, split=1, plot_data=0)
print('Loaded data. Input size is: ', accel_input.shape, gyro_input.shape)


# Phase 2- Create first layer, insert and view spikes:
print("Generating parameter arrays- (Same for acceleration and gyroscope)")
weights_X, weights_Y, LP_array, LF, Theta = generate_and_plot_parameter_arrays(
    plot_param=0, grid_size=GRID_SIZE, theta=THETA, lp_range=LP_RANGE, lf=LF_CONST, w_amplifier=3)

# Create neuron grids
print("Creating neuron grids...")
neuron_grid_accel = create_neuron_grid(weights_X, weights_Y, LP_array, LF, Theta, threshold_pulse=2)
neuron_grid_gyro = create_neuron_grid(weights_X, weights_Y, LP_array, LF, Theta, threshold_pulse=2)

w_amp_accel = 2/(GRID_SIZE*GRID_SIZE)
w_amp_gyro = 2/(GRID_SIZE*GRID_SIZE)
next_layer = create_next_layer(GRID_SIZE,
                               w_amp_accel=w_amp_accel,
                               w_amp_gyro=w_amp_gyro,
                               LF_layer_2=LF_layer_2,
                               LP_LAYER_2=LP_LAYER_2,
                               THETA_LAYER_2=THETA_LAYER_2,
                               threshold_pulse=THRESHOLD,
                               plot=0)

accel_spikes = calculate_grid_spikes(neuron_grid_accel, accel_input)
gyro_spikes = calculate_grid_spikes(neuron_grid_gyro, gyro_input)
combined_spikes = combine_accel_gyro_spikes(accel_spikes, gyro_spikes)


output_spikes = process_next_layer(next_layer, combined_spikes)
save_output_spikes(output_spikes, LF_layer_2=LF_layer_2,LP_LAYER_2=LP_LAYER_2,THETA_LAYER_2=THETA_LAYER_2,THRESHOLD=THRESHOLD)
plot_output_neuron_spikes(output_spikes)
final_position = plot_drone_movement(output_spikes,samples_window=500)   # First try
# final_position = calculate_drone_trajectory(output_spikes) # Second try