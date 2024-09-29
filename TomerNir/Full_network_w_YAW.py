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


def create_YAW_grid(weights_X, LP_array, LF, Theta, threshold_pulse=6):
    neuron_grid = np.empty(GRID_SIZE, dtype=object)
    edge = GRID_SIZE // 2
    for j in range(-edge, edge):
        neuron = SCTNeuron(
            synapses_weights=np.array([weights_X[edge, j + edge]]),
            leakage_factor=LF,
            leakage_period=int(LP_array[edge, j + edge]),
            theta=Theta,
            activation_function=1,  # Using BINARY activation
            threshold_pulse=threshold_pulse,
            log_membrane_potential=True
        )
        neuron_grid[j + edge] = neuron
    return neuron_grid


def calculate_grid_spikes(neuron_grid, input_data):
    """
    Calculate spikes for each neuron in the grid for all input samples.
    """
    if neuron_grid.ndim == 2:  # For accelerometer grid
        GRID_SIZE = neuron_grid.shape[0]
        num_samples = input_data.shape[0]
        spike_array = np.zeros((GRID_SIZE * GRID_SIZE, num_samples))

        for sample_idx in tqdm(range(num_samples), desc="Processing samples"):
            x, y = input_data[sample_idx]
            input_sample = np.array([x, y])

            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    neuron = neuron_grid[i, j]
                    spike = neuron.ctn_cycle(input_sample, True)
                    spike_array[i * GRID_SIZE + j, sample_idx] = 1 if spike else 0

    else:  # For YAW line grid
        GRID_SIZE = neuron_grid.shape[0]
        num_samples = input_data.shape[0]
        spike_array = np.zeros((GRID_SIZE, num_samples))

        for sample_idx in tqdm(range(num_samples), desc="Processing YAW samples"):
            z = input_data[sample_idx]
            input_sample = z

            for i in range(GRID_SIZE):
                neuron = neuron_grid[i]
                spike = neuron.ctn_cycle(input_sample, True)
                spike_array[i, sample_idx] = 1 if spike else 0

    return spike_array


def combine_accel_gyro_spikes(accel_spikes, gyro_spikes):
    """
    Combine spike arrays from accelerometer and gyroscope grids into a single array.
    """
    assert accel_spikes.shape[1] == gyro_spikes.shape[1], "Mismatch in number of samples between accel and gyro data"
    combined_spikes = np.vstack([accel_spikes, gyro_spikes])
    print(f"Combined spike array shape: {combined_spikes.shape}")
    return combined_spikes


def create_next_layer(grid_size=50, next_layer_size=6, plot=0, w_amp_accel=1, w_amp_gyro=1, LF_layer_2=0, LP_LAYER_2=0,
                      THETA_LAYER_2=0, threshold_pulse=0):
    """
    Create the next layer of neurons (4 accel + 2 gyro = 6 neurons).
    """
    accel_length = grid_size * grid_size
    yaw_length = grid_size
    full_length = accel_length + yaw_length

    weights = np.zeros((next_layer_size, full_length))

    # Accel neurons (4)
    weights[0, :accel_length // 2] = w_amp_accel  # Forward (top half)
    weights[1, accel_length // 2:accel_length] = w_amp_accel  # Backward (bottom half)

    # Right (right half of each row)
    row_pattern = np.concatenate([np.zeros(grid_size // 2), np.ones(grid_size // 2)])
    weights[2, :accel_length] = np.tile(row_pattern, grid_size) * w_amp_accel

    # Left (left half of each row)
    row_pattern = np.concatenate([np.ones(grid_size // 2), np.zeros(grid_size // 2)])
    weights[3, :accel_length] = np.tile(row_pattern, grid_size) * w_amp_accel

    # Gyro neurons (2)
    weights[4, accel_length:accel_length + yaw_length // 2] = w_amp_gyro  # CCW rotation (left half of YAW)
    weights[5, accel_length + yaw_length // 2:] = w_amp_gyro  # CW rotation (right half of YAW)

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
        plot_neuron_weights(weights, grid_size)

    return next_layer_neurons


def plot_neuron_weights(weights, grid_size=50):
    fig, axs = plt.subplots(2, 3, figsize=(16, 12))
    fig.suptitle('Input Weights for 6 Neurons', fontsize=12.8)

    cmap_accel = LinearSegmentedColormap.from_list("custom_accel", ["white", "red"])
    cmap_gyro = LinearSegmentedColormap.from_list("custom_gyro", ["white", "green"])

    for i in range(6):
        row = i // 3
        col = i % 3

        if i < 4:  # Acceleration neurons
            weights_to_plot = weights[i, :grid_size * grid_size].reshape(grid_size, grid_size)
            title = f'Accel Neuron {i + 1}'
            cmap = cmap_accel
        else:  # Gyroscope neurons
            weights_to_plot = weights[i, grid_size * grid_size:].reshape(1, -1)
            title = f'Gyro Neuron {i - 3}'
            cmap = cmap_gyro

        im = axs[row, col].imshow(weights_to_plot, cmap=cmap, aspect='auto')
        axs[row, col].set_title(title, fontsize=9.6)

        axs[row, col].set_xticks(np.arange(0, weights_to_plot.shape[1], 10))
        axs[row, col].set_yticks(np.arange(0, weights_to_plot.shape[0], 10))
        axs[row, col].set_xticklabels(np.arange(0, weights_to_plot.shape[1], 10), fontsize=8)
        axs[row, col].set_yticklabels(np.arange(0, weights_to_plot.shape[0], 10), fontsize=8)

        cbar = plt.colorbar(im, ax=axs[row, col], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()


def process_next_layer(next_layer_neurons, combined_spikes):
    """
    Process the combined spikes through the next layer of neurons with a progress bar.
    """
    num_neurons = len(next_layer_neurons)
    num_samples = combined_spikes.shape[1]
    output_spikes = np.zeros((num_neurons, num_samples))

    with tqdm(total=num_samples, desc="Processing next layer") as pbar:
        for t in range(num_samples):
            input_spikes = combined_spikes[:, t]
            for i, neuron in enumerate(next_layer_neurons):
                spike = neuron.ctn_cycle(input_spikes, True)
                output_spikes[i, t] = 1 if spike else 0
            pbar.update(1)

    return output_spikes


def plot_drone_movement(output_spikes, samples_window=10, step_size=0.1):
    """
    Plot the movement of the drone based on output spikes from the neural network.
    """
    num_neurons, num_samples = output_spikes.shape
    num_steps = num_samples // samples_window

    # Define movement directions for each neuron
    directions = [
        (1, 0),  # Accel 1: Forward
        (-1, 0),  # Accel 2: Backward
        (0, 1),  # Accel 3: Right
        (0, -1),  # Accel 4: Left
        (0, 0),  # Gyro 1: Rotate CCW (no linear movement)
        (0, 0)  # Gyro 2: Rotate CW (no linear movement)
    ]

    # Initialize drone position and orientation
    position = np.zeros((num_steps + 1, 2))
    orientation = 0  # in radians

    for step in range(num_steps):
        start_sample = step * samples_window
        end_sample = (step + 1) * samples_window

        # Count spikes in the current window for each neuron
        spike_counts = np.sum(output_spikes[:, start_sample:end_sample], axis=1)

        # Update orientation based on gyro neurons
        orientation += (spike_counts[4] - spike_counts[5]) * np.pi / 18  # 10 degrees per spike

        # Calculate movement
        movement = np.zeros(2)
        for i in range(4):  # Only consider accel neurons for movement
            movement += np.array(directions[i]) * spike_counts[i]

        # Rotate movement based on current orientation
        rotated_movement = np.array([
            movement[0] * np.cos(orientation) - movement[1] * np.sin(orientation),
            movement[0] * np.sin(orientation) + movement[1] * np.cos(orientation)
        ])

        # Update position
        position[step + 1] = position[step] + rotated_movement * step_size

    # Plot the movement
    plt.figure(figsize=(10, 10))
    plt.plot(position[:, 0], position[:, 1], 'b-')
    plt.plot(position[0, 0], position[0, 1], 'go', markersize=10, label='Start')
    plt.plot(position[-1, 0], position[-1, 1], 'ro', markersize=10, label='End')

    # Plot orientation arrows
    for i in range(0, num_steps, num_steps // 20):  # Plot 20 arrows along the path
        dx = np.cos(orientation) * 0.5
        dy = np.sin(orientation) * 0.5
        plt.arrow(position[i, 0], position[i, 1], dx, dy, head_width=0.2, head_length=0.3, fc='r', ec='r')

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
    Plot the spikes of all 6 output neurons in one window.
    """
    num_neurons, num_samples = output_spikes.shape
    time = np.arange(num_samples) * time_window

    plt.figure(figsize=(20, 12))
    for i in range(num_neurons):
        plt.subplot(6, 1, i + 1)
        plt.plot(time, output_spikes[i], drawstyle='steps-post', linewidth=1)
        plt.ylabel(f'Neuron {i + 1}')
        plt.ylim(-0.1, 1.1)
        if i < 5:
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        if i == 5:
            plt.xlabel('Time (s)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.title(f'{"Accel" if i < 4 else "Gyro"} Neuron {i + 1 if i < 4 else i - 3} Spike Activity', fontsize=10)

    plt.tight_layout()
    plt.suptitle('Output Spikes of All 6 Neurons', fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()


def save_output_spikes(output_spikes, LF_layer_2, LP_LAYER_2, THETA_LAYER_2, THRESHOLD):
    """
    Save the output spikes array to a file in the specified path with a unique filename.
    """
    base_path = r'C:\Users\user1\PycharmProjects\SNN-SCTN\Phase_3\output_spikes_arrays'
    os.makedirs(base_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_spikes_LF{LF_layer_2}_LP{LP_LAYER_2}_THETA{THETA_LAYER_2}_THRESH{THRESHOLD}_{timestamp}.npy"
    full_path = os.path.join(base_path, filename)
    np.save(full_path, output_spikes)
    print(f"Output spikes saved to {full_path}")


def calculate_drone_trajectory(output_spikes, samples_window=10, linear_step=0.1, angular_step=np.pi / 18):
    """
    Calculate and plot the trajectory of the drone based on output spikes.
    """
    num_neurons, num_samples = output_spikes.shape
    num_steps = num_samples // samples_window

    # Initialize drone state
    position = np.zeros((num_steps + 1, 2))
    orientation = 0  # in radians

    # Define neuron indices for angular and linear velocities
    angular_neurons = [4, 5]  # CCW and CW rotation
    linear_neurons = [0, 1, 2, 3]  # Forward, Backward, Right, Left

    for step in range(num_steps):
        start_sample = step * samples_window
        end_sample = (step + 1) * samples_window

        # Count spikes in the current window for each neuron
        spike_counts = np.sum(output_spikes[:, start_sample:end_sample], axis=1)

        # Calculate angular velocity
        angular_velocity = (spike_counts[4] - spike_counts[5]) * angular_step
        orientation += angular_velocity

        # Calculate linear velocity
        forward_velocity = (spike_counts[0] - spike_counts[1]) * linear_step
        lateral_velocity = (spike_counts[2] - spike_counts[3]) * linear_step

        # Update position based on orientation and velocities
        position[step + 1, 0] = position[step, 0] + (forward_velocity * np.cos(orientation) - lateral_velocity * np.sin(orientation))
        position[step + 1, 1] = position[step, 1] + (forward_velocity * np.sin(orientation) + lateral_velocity * np.cos(orientation))

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

# Main execution
if __name__ == "__main__":
    dir_path = r'C:\Users\user1\PycharmProjects\SNN-SCTN\Phase_2\Uzh_datasets\indoor_forward_9_davis'
    ts, gyro, accel = breakdown_data(dir_path, start=27264,
                                     stop=56032,
                                     plot_data=0,
                                     fix_axis_and_g=1,
                                     fix_meas=0,
                                     no_Z_accel=0,
                                     no_Z_gyro=0)
    gt_ts, gt_t_xyz, gt_q_wxyz, gt_data, all_translated_points = get_gt_data(dir_path,plot_data=0)
    accel_input, gyro_input = process_accel_gyro_data(accel, gyro, split=1, plot_data=0)  # Only use gyro Z-axis
    print('Loaded data. Input size is: ', accel_input.shape, gyro_input.shape)

    print("Generating parameter arrays")
    weights_X, weights_Y, LP_array, LF, Theta = generate_and_plot_parameter_arrays(
        plot_param=1, grid_size=GRID_SIZE, theta=THETA, lp_range=LP_RANGE, lf=LF_CONST, w_amplifier=3)

    print("Creating neuron grids...")
    neuron_grid_accel = create_neuron_grid(weights_X, weights_Y, LP_array, LF, Theta, threshold_pulse=2)
    neuron_grid_YAW = create_YAW_grid(weights_X, LP_array, LF, Theta, threshold_pulse=2)

    w_amp_accel = 2/(GRID_SIZE*GRID_SIZE)
    w_amp_gyro = 2/GRID_SIZE
    next_layer = create_next_layer(GRID_SIZE,
                                   next_layer_size=6,
                                   w_amp_accel=w_amp_accel,
                                   w_amp_gyro=w_amp_gyro,
                                   LF_layer_2=LF_layer_2,
                                   LP_LAYER_2=LP_LAYER_2,
                                   THETA_LAYER_2=THETA_LAYER_2,
                                   threshold_pulse=THRESHOLD,
                                   plot=0)

    accel_spikes = calculate_grid_spikes(neuron_grid_accel, accel_input)
    gyro_spikes = calculate_grid_spikes(neuron_grid_YAW, gyro_input)
    combined_spikes = combine_accel_gyro_spikes(accel_spikes, gyro_spikes)

    output_spikes = process_next_layer(next_layer, combined_spikes)
    save_output_spikes(output_spikes, LF_layer_2=LF_layer_2, LP_LAYER_2=LP_LAYER_2, THETA_LAYER_2=THETA_LAYER_2, THRESHOLD=THRESHOLD)
    plot_output_neuron_spikes(output_spikes)
    final_position = calculate_drone_trajectory(output_spikes, samples_window=500)

    print("Processing complete. Check the generated plots and saved output spikes.")