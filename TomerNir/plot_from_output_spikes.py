import numpy as np
import matplotlib.pyplot as plt
import os
import glob

plt.switch_backend('Qt5Agg')
from Phase_2.utils_phase2 import breakdown_data, get_gt_data, generate_and_plot_parameter_arrays


def load_output_spikes(file_path):
    """
    Load the output spikes array from a specified file path.

    Parameters:
    file_path (str): Full path of the file to load

    Returns:
    numpy.ndarray: Loaded output spikes array
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at {file_path}")

    # Load the file
    output_spikes = np.load(file_path)
    print(f"Output spikes loaded from {file_path}")

    return output_spikes


def plot_output_neuron_spikes(output_spikes, time_window=0.01):
    """
    Plot the spikes of all 9 output neurons in one window with 10-second x-axis jumps.

    Parameters:
    output_spikes (numpy.ndarray): Array of output spikes with shape (9, num_samples)
    time_window (float): Time step between samples in seconds (default is 0.01s)
    """
    num_neurons, num_samples = output_spikes.shape
    time = np.arange(num_samples) * time_window

    plt.figure(figsize=(20, 12))
    for i in range(num_neurons):
        ax = plt.subplot(9, 1, i + 1)
        plt.plot(time, output_spikes[i], drawstyle='steps-post', linewidth=1)
        plt.ylabel(f'Neuron {i}')
        plt.ylim(-0.1, 1.1)

        # Set x-axis ticks to 10-second intervals
        max_time = time[-1]
        ax.set_xticks(np.arange(0, max_time + 10, 10))
        ax.set_xticklabels([f'{int(t)}s' for t in np.arange(0, max_time + 10, 10)])

        if i < 8:
            plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
        if i == 8:
            plt.xlabel('Time (s)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.title(f'Neuron {i} Spike Activity', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


def plot_drone_movement_w_heading(output_spikes, samples_window, step_size=0.1, start_point=(0, 0), ax=None,
                                  time_window=0.01):
    num_neurons, num_samples = output_spikes.shape
    num_steps = num_samples // samples_window

    accel_directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Left, Up, Right, Down
    gyro_changes = {0: -0.1, 2: -0.1, 6: 0.1, 8: 0.1}  # Counterclockwise: -1, Clockwise: 1

    position = np.zeros((num_steps + 1, 2))
    position[0] = start_point  # Set the starting point
    heading = 90  # Initial heading (in degrees)

    accel_neurons = [1, 3, 5, 7]
    gyro_neurons = [0, 2, 6, 8]

    for step in range(num_steps):
        start_sample = step * samples_window
        end_sample = (step + 1) * samples_window
        spike_counts = np.sum(output_spikes[:, start_sample:end_sample], axis=1)

        # Acceleration decision
        accel_spike_counts = spike_counts[accel_neurons]
        dominant_accel = np.argmax(accel_spike_counts)
        accel_movement = np.array(accel_directions[dominant_accel]) * step_size

        # Gyroscope decision
        gyro_spike_counts = spike_counts[gyro_neurons]
        active_gyros = [neuron for neuron, count in zip(gyro_neurons, gyro_spike_counts) if count > 0]

        # Update heading based on active gyro neurons
        heading_change = sum(gyro_changes.get(neuron, 0) for neuron in active_gyros)
        heading += heading_change
        heading %= 360  # Keep heading between 0 and 359 degrees

        # Calculate movement based on heading
        accel_move = np.array([
            accel_movement[0] * np.cos(np.radians(heading)) - accel_movement[1] * np.sin(np.radians(heading)),
            accel_movement[0] * np.sin(np.radians(heading)) + accel_movement[1] * np.cos(np.radians(heading))
        ])

        # Add gyro step if there's a heading change
        if heading_change != 0:
            gyro_step = np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading))]) * step_size
        else:
            gyro_step = np.zeros(2)

        # Combine acceleration and gyro movements
        total_movement = accel_move + gyro_step

        position[step + 1] = position[step] + total_movement

    ax.plot(position[:, 0], position[:, 1], label=f'Predicted (Window: {samples_window})')
    ax.plot(position[0, 0], position[0, 1], 'go', markersize=6)
    ax.plot(position[-1, 0], position[-1, 1], 'ro', markersize=6)

    # Add markers every 10 seconds
    total_time = num_samples * time_window
    marker_interval = 10  # seconds
    num_markers = int(total_time // marker_interval)

    for i in range(1, num_markers + 1):
        marker_time = i * marker_interval
        marker_step = int(marker_time / (samples_window * time_window))
        if marker_step < len(position):
            ax.plot(position[marker_step, 0], position[marker_step, 1], 'bo', markersize=4)
            ax.annotate(f'{i * 10}s', (position[marker_step, 0], position[marker_step, 1]),
                        textcoords="offset points", xytext=(0, 10), ha='center', fontsize=6)

    return position, position[-1]

def plot_drone_movement(output_spikes, samples_window, step_size=0.1, start_point=(0, 0), ax=None, time_window=0.01):
    num_neurons, num_samples = output_spikes.shape
    num_steps = num_samples // samples_window

    accel_directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Left, Up, Right, Down
    gyro_directions = {0: (-1, 1), 2: (-1, -1), 6: (1, 1), 8: (1, -1)}  # New gyro impacts
    # accel_directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Left, Up, Right, Down
    # gyro_directions = {0: (1, -1), 2: (1, 1), 6: (-1, -1), 8: (-1, 1)}  # New gyro impacts

    position = np.zeros((num_steps + 1, 2))
    position[0] = start_point  # Set the starting point

    accel_neurons = [1, 3, 5, 7]
    gyro_neurons = [0, 2, 6, 8]

    for step in range(num_steps):
        start_sample = step * samples_window
        end_sample = (step + 1) * samples_window
        spike_counts = np.sum(output_spikes[:, start_sample:end_sample], axis=1)

        # Acceleration decision
        accel_spike_counts = spike_counts[accel_neurons]
        dominant_accel = np.argmax(accel_spike_counts)
        accel_movement = np.array(accel_directions[dominant_accel]) * step_size

        # Gyroscope decision
        gyro_spike_counts = spike_counts[gyro_neurons]
        dominant_gyro = gyro_neurons[np.argmax(gyro_spike_counts)]
        gyro_movement = np.array(gyro_directions[dominant_gyro]) * step_size

        # Combine acceleration and gyro movements
        total_movement = accel_movement + gyro_movement

        position[step + 1] = position[step] + total_movement

    ax.plot(position[:, 0], position[:, 1], label=f'Predicted (Window: {samples_window})')
    ax.plot(position[0, 0], position[0, 1], 'go', markersize=6)
    ax.plot(position[-1, 0], position[-1, 1], 'ro', markersize=6)

    # Add markers every 10 seconds
    total_time = num_samples * time_window
    marker_interval = 10  # seconds
    num_markers = int(total_time // marker_interval)

    for i in range(1, num_markers + 1):
        marker_time = i * marker_interval
        marker_step = int(marker_time / (samples_window * time_window))
        if marker_step < len(position):
            ax.plot(position[marker_step, 0], position[marker_step, 1], 'bo', markersize=4)
            ax.annotate(f'{i*10}s', (position[marker_step, 0], position[marker_step, 1]),
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=6)

    return position, position[-1]

# Main execution
if __name__ == "__main__":
    spikes_path = r'C:\Users\user1\PycharmProjects\SNN-SCTN\Phase_3\output_spikes_arrays\output_spikes_LF1_LP50_THETA0_THRESH1_20240911_174918.npy'
    output_spikes = load_output_spikes(spikes_path)
    plot_output_neuron_spikes(output_spikes)
    dir_path = r'C:\Users\user1\PycharmProjects\SNN-SCTN\Phase_2\Uzh_datasets\indoor_forward_9_davis'
    gt_ts, gt_t_xyz, gt_q_wxyz, gt_data, all_translated_points = get_gt_data(dir_path, plot_data=0)

    sample_windows = [500 + i * 200 for i in range(1)]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(sample_windows)))

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(20, 15))
    axs = axs.flatten()

    gt_start_point = all_translated_points[0, :2]  # Use only x and y coordinates

    for idx, (samples_window, color) in enumerate(zip(sample_windows, colors)):
        ax = axs[idx]
        position, final_position = plot_drone_movement(output_spikes, samples_window=samples_window, step_size=1,
                                                       start_point=gt_start_point, ax=ax, time_window=0.01)
        print(f"Final position (window {samples_window}): {final_position}")

        # Plot ground truth using translated points
        ax.plot(all_translated_points[:, 0], all_translated_points[:, 1], 'k--', linewidth=2, label='Ground Truth')
        ax.plot(all_translated_points[0, 0], all_translated_points[0, 1], 'go', markersize=8, label='Start')
        ax.plot(all_translated_points[-1, 0], all_translated_points[-1, 1], 'ro', markersize=8, label='End')

        # Add markers every 10 seconds for ground truth
        total_time = len(all_translated_points) * 0.01  # Assuming 0.01s time step
        marker_interval = 10  # seconds
        num_markers = int(total_time // marker_interval)

        for i in range(1, num_markers + 1):
            marker_time = i * marker_interval
            marker_index = int(marker_time / 0.01)
            if marker_index < len(all_translated_points):
                ax.plot(all_translated_points[marker_index, 0], all_translated_points[marker_index, 1], 'ko', markersize=4)
                ax.annotate(f'{i*20}s', (all_translated_points[marker_index, 0], all_translated_points[marker_index, 1]),
                            textcoords="offset points", xytext=(0,10), ha='center', fontsize=6)

        ax.set_title(f'Window: {samples_window}')
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.grid(True)
        ax.legend()
        ax.axis('equal')

    # Remove any unused subplots
    for idx in range(len(sample_windows), len(axs)):
        fig.delaxes(axs[idx])

    plt.tight_layout()
    plt.show()

