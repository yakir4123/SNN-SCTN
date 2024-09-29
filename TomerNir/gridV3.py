import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from Phase_2.poission_encoding import encode_and_combine
from snn.spiking_neuron import SCTNeuron
from Phase_2.utils_phase2 import breakdown_data, get_gt_data, generate_and_plot_parameter_arrays
from tqdm import tqdm

plt.switch_backend('Qt5Agg')

# Constants
GRID_SIZE = 50
THETA = -0.45
LP_RANGE = (50, 2)
LF_CONST = 1
NUM_GT_SAMPLES_PER_UPDATE = 100
NUM_IMU_SAMPLES_PER_UPDATE = 200
UPDATE_INTERVAL = 500  # in milliseconds, increased for smoother animation

def create_neuron_grid(weights_X, weights_Y, LP_array, LF, Theta, threshold_pulse=6):
    neuron_grid = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
    edge = GRID_SIZE // 2
    for i in range(-edge, edge):
        for j in range(-edge, edge):
            neuron = SCTNeuron(
                synapses_weights=np.array([weights_X[i + edge, j + edge], weights_Y[i + edge, j + edge]]),
                leakage_factor=LF,
                leakage_period=int(LP_array[i + edge, j + edge]),
                theta=Theta,
                activation_function=1,  # Using BINARY activation
                threshold_pulse=threshold_pulse,
                log_membrane_potential=True
            )
            neuron_grid[i + edge, j + edge] = neuron
    return neuron_grid

def update_grid(neuron_grid, input_sample):
    spike_map = np.zeros((GRID_SIZE, GRID_SIZE))
    x, y = input_sample  # Now x and y are single values, not separate positive and negative

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            neuron = neuron_grid[i, j]
            input_data = np.array([x, y])  # Use the combined x and y directly
            spike = neuron.ctn_cycle(input_data, True)
            spike_map[i, j] = 1 if spike else 0
    return spike_map

def pre_calculate_data(accel_x, accel_y, gyro_x, gyro_y,
                       all_translated_points, neuron_grid_accel, neuron_grid_gyro):
    num_frames = len(all_translated_points) // NUM_GT_SAMPLES_PER_UPDATE + 1

    spike_avg_accel_frames = []
    spike_avg_gyro_frames = []
    gt_trajectory_frames = []
    accel_windows = []
    gyro_windows = []

    print("Pre-calculating data...")
    for frame in tqdm(range(num_frames), desc="Processing frames", unit="frame"):
        gt_start_frame = frame * NUM_GT_SAMPLES_PER_UPDATE
        gt_end_frame = min((frame + 1) * NUM_GT_SAMPLES_PER_UPDATE, len(all_translated_points))

        imu_start_frame = frame * NUM_IMU_SAMPLES_PER_UPDATE
        imu_end_frame = min((frame + 1) * NUM_IMU_SAMPLES_PER_UPDATE, len(accel_x))

        # Calculate spike averages
        spike_sum_accel = np.zeros((GRID_SIZE, GRID_SIZE))
        spike_sum_gyro = np.zeros((GRID_SIZE, GRID_SIZE))
        for i in range(imu_start_frame, imu_end_frame):
            accel_sample = [accel_x[i], accel_y[i]]
            gyro_sample = [gyro_x[i], gyro_y[i]]
            spike_map_accel = update_grid(neuron_grid_accel, accel_sample)
            spike_map_gyro = update_grid(neuron_grid_gyro, gyro_sample)
            spike_sum_accel += spike_map_accel
            spike_sum_gyro += spike_map_gyro

        spike_avg_accel = spike_sum_accel / (imu_end_frame - imu_start_frame)
        spike_avg_gyro = spike_sum_gyro / (imu_end_frame - imu_start_frame)

        spike_avg_accel_frames.append(spike_avg_accel)
        spike_avg_gyro_frames.append(spike_avg_gyro)

        # Store GT trajectory
        gt_trajectory_frames.append(all_translated_points[:gt_end_frame])

        # Store IMU windows
        accel_windows.append([accel_x[imu_start_frame:imu_end_frame],
                              accel_y[imu_start_frame:imu_end_frame]])
        gyro_windows.append([gyro_x[imu_start_frame:imu_end_frame],
                             gyro_y[imu_start_frame:imu_end_frame]])

    return spike_avg_accel_frames, spike_avg_gyro_frames, gt_trajectory_frames, accel_windows, gyro_windows

def setup_visualization(accel, gyro, all_translated_points):
    global fig, ax_accel, ax_gyro, ax_traj, ax_accel_plot, ax_gyro_plot
    global im_accel, im_gyro, drone_plot, trajectory_plot, index_text, accel_scatter, gyro_scatter
    global accel_raw_scatter, gyro_raw_scatter

    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])

    # Translated plot (top row, center)
    ax_traj = fig.add_subplot(gs[0, :])
    drone_plot = ax_traj.scatter([], [], c='r', s=50, animated=True)
    trajectory_plot, = ax_traj.plot([], [], 'b-', alpha=0.5, animated=True)
    ax_traj.set_xlim(all_translated_points[:, 0].min(), all_translated_points[:, 0].max())
    ax_traj.set_ylim(all_translated_points[:, 1].min(), all_translated_points[:, 1].max())
    ax_traj.set_title("Drone Movement (2D)", fontsize=14)
    ax_traj.set_xlabel("X")
    ax_traj.set_ylabel("Y")

    # Accelerometer neuron grid (middle row, left)
    ax_accel = fig.add_subplot(gs[1, 0])
    cmap_accel = LinearSegmentedColormap.from_list("", ["white", "red"])
    im_accel = ax_accel.imshow(np.zeros((GRID_SIZE, GRID_SIZE)), cmap=cmap_accel, animated=True, vmin=0, vmax=1, aspect='equal')
    fig.colorbar(im_accel, ax=ax_accel, shrink=0.8)
    ax_accel.set_title("Accelerometer Neuron Grid", fontsize=14)

    # Acceleration input plot (middle row, right)
    ax_accel_plot = fig.add_subplot(gs[1, 1])
    accel_scatter = ax_accel_plot.scatter([], [], c='b', s=5)
    accel_raw_scatter = ax_accel_plot.scatter([], [], c='r', s=5, alpha=0.3)
    ax_accel_plot.set_xlim(-2, 2)
    ax_accel_plot.set_ylim(-2, 2)
    ax_accel_plot.set_aspect('equal', adjustable='box')
    ax_accel_plot.set_title("Acceleration Input", fontsize=14)
    ax_accel_plot.set_xlabel("X-Acceleration")
    ax_accel_plot.set_ylabel("Y-Acceleration")

    # Gyroscope neuron grid (bottom row, left)
    ax_gyro = fig.add_subplot(gs[2, 0])
    cmap_gyro = LinearSegmentedColormap.from_list("", ["white", "green"])
    im_gyro = ax_gyro.imshow(np.zeros((GRID_SIZE, GRID_SIZE)), cmap=cmap_gyro, animated=True, vmin=0, vmax=1, aspect='equal')
    fig.colorbar(im_gyro, ax=ax_gyro, shrink=0.8)
    ax_gyro.set_title("Gyroscope Neuron Grid", fontsize=14)

    # Gyroscope input plot (bottom row, right)
    ax_gyro_plot = fig.add_subplot(gs[2, 1])
    gyro_scatter = ax_gyro_plot.scatter([], [], c='g', s=5)
    gyro_raw_scatter = ax_gyro_plot.scatter([], [], c='r', s=5, alpha=0.5)
    ax_gyro_plot.set_xlim(-2, 2)
    ax_gyro_plot.set_ylim(-2, 2)
    ax_gyro_plot.set_aspect('equal', adjustable='box')
    ax_gyro_plot.set_title("Gyroscope Input", fontsize=14)
    ax_gyro_plot.set_xlabel("X-Gyroscope")
    ax_gyro_plot.set_ylabel("Y-Gyroscope")

    # Add text to display current index
    index_text = ax_traj.text(0.02, 0.98, "", transform=ax_traj.transAxes, verticalalignment='top', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

def animate(frame):
    global im_accel, im_gyro, drone_plot, trajectory_plot, index_text, accel_scatter, gyro_scatter
    global accel_raw_scatter, gyro_raw_scatter
    global spike_avg_accel_frames, spike_avg_gyro_frames, gt_trajectory_frames, accel_windows, gyro_windows
    global accel_raw_windows, gyro_raw_windows

    # Update the spike heatmaps
    im_accel.set_array(spike_avg_accel_frames[frame])
    im_gyro.set_array(spike_avg_gyro_frames[frame])

    # Update GT plot
    current_trajectory = gt_trajectory_frames[frame]
    drone_plot.set_offsets(current_trajectory[-1:, :2])
    trajectory_plot.set_data(current_trajectory[:, 0], current_trajectory[:, 1])

    # Update acceleration plot
    accel_window = accel_windows[frame]
    accel_scatter.set_offsets(np.column_stack((accel_window[0], accel_window[1])))
    accel_raw_window = accel_raw_windows[frame]
    accel_raw_scatter.set_offsets(np.column_stack((accel_raw_window[0], accel_raw_window[1])))

    # Update gyroscope plot
    gyro_window = gyro_windows[frame]
    gyro_scatter.set_offsets(np.column_stack((gyro_window[0], gyro_window[1])))
    gyro_raw_window = gyro_raw_windows[frame]
    gyro_raw_scatter.set_offsets(np.column_stack((gyro_raw_window[0], gyro_raw_window[1])))

    # Update the index text
    gt_end_frame = (frame + 1) * NUM_GT_SAMPLES_PER_UPDATE
    imu_end_frame = (frame + 1) * NUM_IMU_SAMPLES_PER_UPDATE
    index_text.set_text(f"GT Frames: 0 - {gt_end_frame}, IMU Frames: 0 - {imu_end_frame}")

    return im_accel, im_gyro, drone_plot, trajectory_plot, index_text, accel_scatter, gyro_scatter, accel_raw_scatter, gyro_raw_scatter


def main():
    global spike_avg_accel_frames, spike_avg_gyro_frames, gt_trajectory_frames, accel_windows, gyro_windows
    global accel_raw_windows, gyro_raw_windows

    # Load and process data
    dir_path = r'C:\Users\user1\PycharmProjects\SNN-SCTN\Phase_2\Uzh_datasets\indoor_forward_9_davis'
    ts, gyro, accel = breakdown_data(dir_path,
                                     start=27264,
                                     stop=56032,
                                     plot_data=0,
                                     fix_axis_and_g=1,
                                     fix_meas=0,
                                     no_Z_accel=0,
                                     no_Z_gyro=0)
    gt_ts, gt_t_xyz, gt_q_wxyz, gt_data, all_translated_points = get_gt_data(dir_path, plot_data=0)

    # Normalize raw data
    accel_norm = 2 * (accel - accel.min(axis=0)) / (accel.max(axis=0) - accel.min(axis=0)) - 1
    gyro_norm = 2 * (gyro - gyro.min(axis=0)) / (gyro.max(axis=0) - gyro.min(axis=0)) - 1

    # Apply Poisson encoding and combine
    print("Applying Poisson encoding and combining...")
    accel_x_encoded = encode_and_combine(accel[:, 0])
    accel_y_encoded = encode_and_combine(accel[:, 1])
    gyro_x_encoded = encode_and_combine(gyro[:, 0])
    gyro_y_encoded = encode_and_combine(gyro[:, 1])

    print("Generating parameter arrays- (Same for acceleration and gyroscope)")
    weights_X, weights_Y, LP_array, LF, Theta = generate_and_plot_parameter_arrays(
        plot_param=0, grid_size=GRID_SIZE, theta=THETA, lp_range=LP_RANGE, lf=LF_CONST, w_amplifier=3)


    # Create neuron grids
    print("Creating neuron grids...")
    neuron_grid_accel = create_neuron_grid(weights_X, weights_Y, LP_array, LF, Theta, threshold_pulse=2)
    neuron_grid_gyro = create_neuron_grid(weights_X, weights_Y, LP_array, LF, Theta, threshold_pulse=2)

    # Pre-calculate all data with encoded values
    spike_avg_accel_frames, spike_avg_gyro_frames, gt_trajectory_frames, accel_windows, gyro_windows = pre_calculate_data(
        accel_x_encoded, accel_y_encoded, gyro_x_encoded, gyro_y_encoded,
        all_translated_points, neuron_grid_accel, neuron_grid_gyro)

    # Prepare raw data windows
    num_frames = len(gt_trajectory_frames)
    accel_raw_windows = []
    gyro_raw_windows = []
    for frame in range(num_frames):
        start = frame * NUM_IMU_SAMPLES_PER_UPDATE
        end = min((frame + 1) * NUM_IMU_SAMPLES_PER_UPDATE, len(accel_norm))
        accel_raw_windows.append([accel_norm[start:end, 0], accel_norm[start:end, 1]])
        gyro_raw_windows.append([gyro_norm[start:end, 0], gyro_norm[start:end, 1]])

    print("Setting up visualization...")
    setup_visualization(accel, gyro, all_translated_points)

    print("Starting animation...")
    num_frames = len(gt_trajectory_frames)
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=UPDATE_INTERVAL, blit=True)

    plt.show()

if __name__ == "__main__":
    main()