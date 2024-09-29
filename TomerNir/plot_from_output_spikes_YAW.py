import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from Phase_2.utils_phase2 import get_gt_data

plt.switch_backend('Qt5Agg')


def load_output_spikes(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at {file_path}")
    output_spikes = np.load(file_path)
    print(f"Output spikes loaded from {file_path}")
    return output_spikes


def plot_output_neuron_spikes(output_spikes, gt_trajectory, time_window=0.01):
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
        plt.title(f'{"Accel" if i < 4 else "Gyro YAW"} Neuron {i + 1 if i < 4 else i - 3} Spike Activity', fontsize=10)

        # Add vertical line to indicate end of ground truth data
        gt_end_time = len(gt_trajectory) * time_window
        plt.axvline(x=gt_end_time, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.suptitle('Output Spikes of All 6 Neurons', fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()


def calculate_drone_trajectory(output_spikes, start_point, samples_window=10, linear_step=0.1, angular_step=np.pi / 360,
                               angular_movement_step=0.05):
    num_neurons, num_samples = output_spikes.shape
    num_steps = num_samples // samples_window

    position = np.zeros((num_steps + 1, 2))
    position[0] = start_point
    orientation = 0  # in radians

    for step in tqdm(range(num_steps), desc="Calculating trajectory"):
        start_sample = step * samples_window
        end_sample = (step + 1) * samples_window

        spike_counts = np.sum(output_spikes[:, start_sample:end_sample], axis=1)

        # Update orientation based on YAW gyro neurons
        orientation_change =  angular_step
        orientation += orientation_change
        orientation %= 2 * np.pi  # Keep orientation between 0 and 2π

        # Calculate linear movement
        forward_backward = (spike_counts[0] - spike_counts[1]) * linear_step
        right_left = (spike_counts[2] - spike_counts[3]) * linear_step

        # Calculate angular movement
        angular_movement = np.array([0.0, 0.0])
        if spike_counts[4] > 0:  # Counter-clockwise rotation
            angular_movement += np.array([-angular_movement_step, angular_movement_step])
        if spike_counts[5] > 0:  # Clockwise rotation
            angular_movement += np.array([angular_movement_step, -angular_movement_step])

        # Rotate angular movement based on current orientation
        rotated_angular_movement = np.array([
            angular_movement[0] * np.cos(orientation) - angular_movement[1] * np.sin(orientation),
            angular_movement[0] * np.sin(orientation) + angular_movement[1] * np.cos(orientation)
        ])

        # Apply movement based on current orientation
        position[step + 1, 0] = position[step, 0] + forward_backward * np.cos(orientation) - right_left * np.sin(
            orientation) + rotated_angular_movement[0]
        position[step + 1, 1] = position[step, 1] + forward_backward * np.sin(orientation) + right_left * np.cos(
            orientation) + rotated_angular_movement[1]

    return position


import numpy as np
from tqdm import tqdm


def calculate_drone_trajectory_v5(output_spikes, start_point, samples_window=10, step_size=0.1,
                                  max_angular_step=np.pi / 15):
    """
    Calculate the trajectory of the drone based on output spikes, using proportional movement for all neurons.

    :param output_spikes: Array of spike data for all neurons
    :param start_point: Initial position of the drone
    :param samples_window: Number of samples to consider for each step
    :param step_size: Maximum size of each step taken by the drone
    :param max_angular_step: Maximum amount of rotation per sample window
    :return: Array of drone positions
    """
    num_neurons, num_samples = output_spikes.shape
    num_steps = num_samples // samples_window

    position = np.zeros((num_steps + 1, 2))
    position[0] = start_point
    orientation = -np.pi/(4/3)  # in radians

    for step in tqdm(range(num_steps), desc="Calculating trajectory"):
        start_sample = step * samples_window
        end_sample = (step + 1) * samples_window

        spike_counts = np.sum(output_spikes[:, start_sample:end_sample], axis=1)

        pos_y_spikes = spike_counts[0]  # Positive Y acceleration
        neg_y_spikes = spike_counts[1]  # Negative Y acceleration
        pos_x_spikes = spike_counts[2]  # Positive X acceleration
        neg_x_spikes = spike_counts[3]  # Negative X acceleration
        gyro_ccw_spikes = spike_counts[4]  # Counter-clockwise rotation
        gyro_cw_spikes = spike_counts[5]  # Clockwise rotation

        # Calculate proportional movements
        y_movement = (pos_y_spikes - neg_y_spikes) / samples_window * step_size
        x_movement = (pos_x_spikes - neg_x_spikes) / samples_window * step_size
        angular_change = (gyro_ccw_spikes - gyro_cw_spikes) / samples_window * max_angular_step

        # Update orientation
        orientation += angular_change
        orientation %= 2 * np.pi  # Keep orientation between 0 and 2π

        # Calculate movement vector
        movement = np.zeros(2)

        # Forward movement (fusion of X and Y)
        forward_movement = (x_movement + y_movement) / 2
        movement += np.array([np.cos(orientation), np.sin(orientation)]) * forward_movement

        # Sideways movement (difference between X and Y)
        sideways_movement = (x_movement - y_movement) / 2
        movement += np.array([-np.sin(orientation), np.cos(orientation)]) * sideways_movement

        # Gyro-induced movement (always forward in the current orientation)
        gyro_movement = abs(angular_change) / max_angular_step * step_size
        movement += np.array([np.cos(orientation), np.sin(orientation)]) * gyro_movement

        # Apply movement
        position[step + 1] = position[step] + movement

    return position

def calculate_drone_trajectory_v4(output_spikes, start_point, samples_window=10, step_size=0.1,
                                  max_angular_step=np.pi / 15, threshold_percent=20):
    """
    Calculate the trajectory of the drone based on output spikes, using fusion of acceleration neurons and gyro neurons.
    Angular steps are proportional to spike count and always include forward movement.

    :param output_spikes: Array of spike data for all neurons
    :param start_point: Initial position of the drone
    :param samples_window: Number of samples to consider for each step
    :param step_size: Size of each step taken by the drone
    :param max_angular_step: Maximum amount of rotation per sample window
    :param threshold_percent: Percentage of spikes required to trigger a movement for acceleration neurons
    :return: Array of drone positions
    """
    num_neurons, num_samples = output_spikes.shape
    num_steps = num_samples // samples_window

    position = np.zeros((num_steps + 1, 2))
    position[0] = start_point
    orientation = -np.pi/(4/3)  # in radians

    threshold_count = samples_window * threshold_percent / 100

    for step in tqdm(range(num_steps), desc="Calculating trajectory"):
        start_sample = step * samples_window
        end_sample = (step + 1) * samples_window

        spike_counts = np.sum(output_spikes[:, start_sample:end_sample], axis=1)

        pos_y_spikes = spike_counts[0]  # Positive Y acceleration
        pos_x_spikes = spike_counts[2]  # Positive X acceleration
        gyro_ccw_spikes = spike_counts[4]  # Counter-clockwise rotation
        gyro_cw_spikes = spike_counts[5]  # Clockwise rotation

        move_pos_y = pos_y_spikes > threshold_count
        move_pos_x = pos_x_spikes > threshold_count

        # Calculate proportional angular change
        angular_change = (gyro_ccw_spikes - gyro_cw_spikes) / samples_window * max_angular_step
        orientation += angular_change
        orientation %= 2 * np.pi  # Keep orientation between 0 and 2π

        # Determine movement
        movement = np.zeros(2)

        # Acceleration movement
        if move_pos_x and move_pos_y:
            # Fusion of positive X and Y: move forward
            movement += np.array([np.cos(orientation), np.sin(orientation)]) * step_size
        elif move_pos_x:
            # Only positive X: 30 degrees to the right
            angle = orientation - np.pi / 6  # 30 degrees to the right
            movement += np.array([np.cos(angle), np.sin(angle)]) * step_size
        elif move_pos_y:
            # Only positive Y: 30 degrees to the left
            angle = orientation + np.pi / 6  # 30 degrees to the left
            movement += np.array([np.cos(angle), np.sin(angle)]) * step_size

        # Gyro movement (always forward in the current orientation)
        if gyro_ccw_spikes > 0 or gyro_cw_spikes > 0:
            movement += np.array([np.cos(orientation), np.sin(orientation)]) * step_size

        # Apply movement
        position[step + 1] = position[step] + movement

    return position
def calculate_drone_trajectory_v3(output_spikes, start_point, samples_window=10, step_size=0.1,
                                  angular_step=np.pi / 90, threshold_percent=5):
    """
    Calculate the trajectory of the drone based on output spikes, using fusion of acceleration neurons and gyro neurons.
    Angular steps now include forward movement as well.

    :param output_spikes: Array of spike data for all neurons
    :param start_point: Initial position of the drone
    :param samples_window: Number of samples to consider for each step
    :param step_size: Size of each step taken by the drone
    :param angular_step: Amount of rotation per gyro spike
    :param threshold_percent: Percentage of spikes required to trigger a movement
    :return: Array of drone positions
    """
    num_neurons, num_samples = output_spikes.shape
    num_steps = num_samples // samples_window

    position = np.zeros((num_steps + 1, 2))
    position[0] = start_point
    orientation = 0  # in radians

    threshold_count = samples_window * threshold_percent / 100

    for step in tqdm(range(num_steps), desc="Calculating trajectory"):
        start_sample = step * samples_window
        end_sample = (step + 1) * samples_window

        spike_counts = np.sum(output_spikes[:, start_sample:end_sample], axis=1)

        pos_y_spikes = spike_counts[0]  # Positive Y acceleration
        pos_x_spikes = spike_counts[2]  # Positive X acceleration
        gyro_ccw_spikes = spike_counts[4]  # Counter-clockwise rotation
        gyro_cw_spikes = spike_counts[5]  # Clockwise rotation

        move_pos_y = pos_y_spikes > threshold_count
        move_pos_x = pos_x_spikes > threshold_count
        rotate_ccw = gyro_ccw_spikes > threshold_count
        rotate_cw = gyro_cw_spikes > threshold_count

        # Determine movement
        movement = np.zeros(2)
        if move_pos_x and move_pos_y:
            # Fusion of positive X and Y: move forward
            movement = np.array([np.cos(orientation), np.sin(orientation)]) * step_size
        elif move_pos_x:
            # Only positive X: 30 degrees to the right
            angle = orientation - np.pi / 6  # 30 degrees to the right
            movement = np.array([np.cos(angle), np.sin(angle)]) * step_size
        elif move_pos_y:
            # Only positive Y: 30 degrees to the left
            angle = orientation + np.pi / 6  # 30 degrees to the left
            movement = np.array([np.cos(angle), np.sin(angle)]) * step_size

        # Update orientation and add movement for gyro neurons
        if rotate_ccw:
            orientation += angular_step
            movement += np.array([np.cos(orientation), np.sin(orientation)]) * step_size
        elif rotate_cw:
            orientation -= angular_step
            movement += np.array([np.cos(orientation), np.sin(orientation)]) * step_size

        orientation %= 2 * np.pi  # Keep orientation between 0 and 2π

        # Apply movement
        position[step + 1] = position[step] + movement

    return position
def calculate_drone_trajectory_v2(output_spikes, start_point, samples_window=10, step_size=1,
                                  angular_step=np.pi / 180, threshold_percent=20):
    """
    Calculate the trajectory of the drone based on output spikes, using only forward acceleration and gyro neurons.

    :param output_spikes: Array of spike data for all neurons
    :param start_point: Initial position of the drone
    :param samples_window: Number of samples to consider for each step
    :param step_size: Size of each step taken by the drone
    :param angular_step: Amount of rotation per gyro spike
    :param threshold_percent: Percentage of spikes required to trigger a movement
    :return: Array of drone positions
    """
    num_neurons, num_samples = output_spikes.shape
    num_steps = num_samples // samples_window

    position = np.zeros((num_steps + 1, 2))
    position[0] = start_point
    orientation = -np.pi/4  # in radians

    threshold_count = samples_window * threshold_percent / 100

    for step in tqdm(range(num_steps), desc="Calculating trajectory"):
        start_sample = step * samples_window
        end_sample = (step + 1) * samples_window

        spike_counts = np.sum(output_spikes[:, start_sample:end_sample], axis=1)

        forward_spikes = spike_counts[3]  # Assuming forward acceleration is the first neuron
        gyro_ccw_spikes = spike_counts[5]  # Counter-clockwise rotation
        gyro_cw_spikes = spike_counts[4]  # Clockwise rotation

        move_forward = forward_spikes > threshold_count
        rotate_ccw = gyro_ccw_spikes > threshold_count
        rotate_cw = gyro_cw_spikes > threshold_count

        # Update orientation based on gyro neurons
        if rotate_ccw:
            orientation += angular_step
        elif rotate_cw:
            orientation -= angular_step
        orientation %= 2 * np.pi  # Keep orientation between 0 and 2π

        # Determine movement
        if move_forward:
            # Move forward based on current orientation
            position[step + 1, 0] = position[step, 0] + step_size * np.cos(orientation)
            position[step + 1, 1] = position[step, 1] + step_size * np.sin(orientation)
        elif rotate_ccw and not move_forward:
            # Move forward and left
            position[step + 1, 0] = position[step, 0] + step_size * np.cos(orientation + np.pi / 4)
            position[step + 1, 1] = position[step, 1] + step_size * np.sin(orientation + np.pi / 4)
        elif rotate_cw and not move_forward:
            # Move forward and right
            position[step + 1, 0] = position[step, 0] + step_size * np.cos(orientation - np.pi / 4)
            position[step + 1, 1] = position[step, 1] + step_size * np.sin(orientation - np.pi / 4)
        else:
            # No movement, maintain previous position
            position[step + 1] = position[step]

    return position

def plot_trajectory(predicted_trajectory, gt_trajectory, time_window=0.01, samples_window=10):
    plt.figure(figsize=(12, 10))
    plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'b-', label='Predicted')
    plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'r--', label='Ground Truth')
    plt.plot(gt_trajectory[0, 0], gt_trajectory[0, 1], 'go', markersize=10, label='Start')
    plt.plot(predicted_trajectory[-1, 0], predicted_trajectory[-1, 1], 'bo', markersize=10, label='End (Predicted)')
    plt.plot(gt_trajectory[-1, 0], gt_trajectory[-1, 1], 'ro', markersize=10, label='End (Ground Truth)')

    # Add time markers
    total_time = len(gt_trajectory) * time_window
    marker_interval = 10  # seconds
    num_markers = int(total_time // marker_interval)

    for i in range(1, num_markers + 1):
        marker_time = i * marker_interval
        gt_marker_index = int(marker_time / time_window)
        predicted_marker_index = int(gt_marker_index / samples_window)

        if gt_marker_index < len(gt_trajectory) and predicted_marker_index < len(predicted_trajectory):
            plt.plot(gt_trajectory[gt_marker_index, 0], gt_trajectory[gt_marker_index, 1], 'r.', markersize=8)
            plt.plot(predicted_trajectory[predicted_marker_index, 0], predicted_trajectory[predicted_marker_index, 1],
                     'b.', markersize=8)
            plt.annotate(f'{i * 10}s', (
            predicted_trajectory[predicted_marker_index, 0], predicted_trajectory[predicted_marker_index, 1]),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    plt.title('Drone Trajectory: Predicted vs Ground Truth')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    spikes_path = r'C:\Users\user1\PycharmProjects\SNN-SCTN\Phase_3\output_spikes_arrays\output_spikes_LF0.7_LP50_THETA0_THRESH1_20240912_092419.npy'
    output_spikes = load_output_spikes(spikes_path)

    dir_path = r'C:\Users\user1\PycharmProjects\SNN-SCTN\Phase_2\Uzh_datasets\indoor_forward_9_davis'
    _, _, _, _, gt_trajectory = get_gt_data(dir_path, plot_data=0)

    # plot_output_neuron_spikes(output_spikes, gt_trajectory)

    samples_window = 20
    start_point = gt_trajectory[0, :2]  # Use the start point from ground truth
    predicted_trajectory = calculate_drone_trajectory_v4(output_spikes, start_point, step_size=0.15, samples_window=samples_window)

    plot_trajectory(predicted_trajectory, gt_trajectory[:, :2], samples_window=samples_window)

    print("Processing complete. Check the generated plots.")