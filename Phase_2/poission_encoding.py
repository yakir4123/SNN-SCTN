import numpy as np
import matplotlib.pyplot as plt


plt.switch_backend('Qt5Agg')

def  poisson_encoding(data, time_window=0.001, max_rate=1000):
    """
    Encode data using Poisson encoding.

    :param data: Input data to be encoded
    :param time_window: Time window for encoding (in seconds)
    :param max_rate: Maximum firing rate (in Hz)
    :return: Encoded spike train as integers (0 or 1)
    """
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    rates = normalized_data * max_rate
    spike_probabilities = rates * time_window
    return (np.random.rand(*data.shape) < spike_probabilities).astype(int)


def encode_and_combine(data):
    pos = poisson_encoding(np.maximum(data, 0), 0.001, 1000)
    neg = poisson_encoding(np.abs(np.minimum(data, 0)), 0.001, 1000)
    return np.where(pos > 0, pos, -neg)


def process_accel_gyro_data(accel, gyro, split=0, plot_data=0):
    """
    Process acceleration and gyroscope data and return a 2D array of encoded values.

    Parameters:
    accel (numpy.ndarray): Acceleration data with shape (n, 3) or (n, 2)
    gyro (numpy.ndarray): Gyroscope data with shape (n, 3) or (n, 2)

    Returns:
    numpy.ndarray: 2D array with shape (n, 4) where each column is an encoded signal
                   (accel_x, accel_y, gyro_x, gyro_y) and each row is a timestamp
    """
    # Ensure we're only using x and y components
    accel = accel[:, :2]
    gyro = gyro[:, :2]

    # Encode data
    accel_x_encoded = encode_and_combine(accel[:, 0])
    accel_y_encoded = encode_and_combine(accel[:, 1])
    gyro_x_encoded = encode_and_combine(gyro[:, 0])
    gyro_y_encoded = encode_and_combine(gyro[:, 1])

    # Combine all encoded data into a single 2D array
    combined_data = np.column_stack([accel_x_encoded, accel_y_encoded, gyro_x_encoded, gyro_y_encoded])
    if plot_data:
        _plot_poisson_encoded_imu_data(accel,gyro,accel_x_encoded, accel_y_encoded, gyro_x_encoded, gyro_y_encoded)

    if split:
        return np.column_stack([accel_x_encoded, accel_y_encoded]), np.column_stack([gyro_x_encoded, gyro_y_encoded])

    return combined_data


# def plot_processed_accel_gyro_data(combined_data):
#     """
#     Plot the processed accelerometer and gyroscope data in separate subplots within a single window.
#
#     Parameters:
#     combined_data (numpy.ndarray): A 2D array with shape (n, 4) where each column is an encoded signal
#                                    (accel_x, accel_y, gyro_x, gyro_y) and each row is a timestamp
#     """
#     num_samples, num_signals = combined_data.shape
#     signal_names = ['Accelerometer X', 'Accelerometer Y', 'Gyroscope X', 'Gyroscope Y']
#
#     fig, axs = plt.subplots(2, 2, figsize=(15, 10))
#     fig.suptitle('Processed Accelerometer and Gyroscope Data', fontsize=16)
#
#     for i in range(num_signals):
#         row = i // 2
#         col = i % 2
#
#         axs[row, col].plot(combined_data[:, i])
#         axs[row, col].set_title(signal_names[i])
#         axs[row, col].set_xlabel('Time')
#         axs[row, col].set_ylabel('Encoded Value')
#         axs[row, col].grid(True)
#
#     plt.tight_layout()
#     plt.show()

def _visualize_combined_data(accel, gyro, encoded_accel_x, encoded_accel_y, encoded_gyro_x, encoded_gyro_y):
    """
    Visualize original and encoded data for both accelerometer and gyroscope on the same subplots.

    :param accel: Original accelerometer data
    :param gyro: Original gyroscope data
    :param encoded_accel: Encoded accelerometer data
    :param encoded_gyro: Encoded gyroscope data
    """
    fig, axs = plt.subplots(4, 1, figsize=(15, 20))
    fig.suptitle("IMU Data: Original vs Poisson Encoded", fontsize=16)

    data_pairs = [
        (accel[:, 0], encoded_accel_x, "Accelerometer X"),
        (accel[:, 1], encoded_accel_y, "Accelerometer Y"),
        (gyro[:, 0], encoded_gyro_x, "Gyroscope X"),
        (gyro[:, 1], encoded_gyro_y, "Gyroscope Y")
    ]

    for i, (original, encoded, title) in enumerate(data_pairs):
        ax = axs[i]

        # Plot encoded data as a background image
        im = ax.imshow(encoded.reshape(1, -1), aspect='auto', cmap='binary', alpha=1,
                       extent=[0, len(original), np.min(original), np.max(original)])

        # Plot original data
        ax.plot(original, color='blue', linewidth=2, label='Original')

        # Set title and labels
        ax.set_title(f"{title}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Value")

        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


# def _plot_poisson_encoded_imu_data(accel, gyro, accel_x_encoded, accel_y_encoded, gyro_x_encoded, gyro_y_encoded,
#                                   time_window=0.01, max_rate=100, threshold=0.2):
#     fig, axs = plt.subplots(2, 2, figsize=(20, 15))
#     fig.suptitle(
#         f'Poisson Encoded IMU Data (time window: {time_window}s, max rate: {max_rate}Hz, threshold: {threshold})',
#         fontsize=16)
#
#     raw_data = [accel[:, 0], accel[:, 1], gyro[:, 0], gyro[:, 1]]
#     encoded_data = [accel_x_encoded, accel_y_encoded, gyro_x_encoded, gyro_y_encoded]
#     titles = ['Accel X', 'Accel Y', 'Gyro X', 'Gyro Y']
#
#     for i, (ax, raw, encoded, title) in enumerate(zip(axs.flat, raw_data, encoded_data, titles)):
#         time = np.arange(len(raw)) * time_window
#
#         # Plot raw data
#         ax.plot(time, raw, color='purple', alpha=0.3, label='Raw Data')
#
#         # Plot encoded spikes
#         positive_mask = encoded > 0
#         negative_mask = encoded < 0
#
#         spike_height = np.max(np.abs(raw)) * 0.2  # Adjust spike height
#         ax.vlines(time[positive_mask], 0, spike_height, color='blue', alpha=0.5, label='Positive Spikes')
#         ax.vlines(time[negative_mask], -spike_height, 0, color='red', alpha=0.5, label='Negative Spikes')
#
#         ax.set_title(title)
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('Value / Spike')
#         ax.legend()
#
#         # Set y-axis limits
#         y_max = max(np.max(np.abs(raw)), 1) * 1.2
#         ax.set_ylim(-y_max, y_max)
#
#         # Add grid
#         ax.grid(True, linestyle='--', alpha=0.3)
#
#     plt.tight_layout()
#     plt.show()

def _plot_poisson_encoded_imu_data(accel, gyro, accel_x_encoded, accel_y_encoded, gyro_x_encoded, gyro_y_encoded,
                                   time_window=0.01, max_rate=100, threshold=0.2):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))


    time = np.arange(len(accel)) * time_window

    # Accelerometer subplot
    ax1.plot(time, accel[:, 0], color='blue', alpha=0.7, label='Accel X')
    ax1.plot(time, accel[:, 1], color='red', alpha=0.7, label='Accel Y')

    ax1.scatter(time[accel_x_encoded > 0], 5 * np.ones_like(time[accel_x_encoded > 0]),
                color='blue', marker='.', s=10, label='Accel X +')
    ax1.scatter(time[accel_x_encoded < 0], -5 * np.ones_like(time[accel_x_encoded < 0]),
                color='blue', marker='.', s=10, label='Accel X -')

    ax1.scatter(time[accel_y_encoded > 0], 5.2 * np.ones_like(time[accel_y_encoded > 0]),
                color='red', marker='.', s=10, label='Accel Y +')
    ax1.scatter(time[accel_y_encoded < 0], -5.2 * np.ones_like(time[accel_y_encoded < 0]),
                color='red', marker='.', s=10, label='Accel Y -')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    ax1.set_title('Accelerometer Data')
    ax1.set_ylabel('Acceleration (m/s²) / Spike')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(10))

    # Set y-axis limits for accelerometer
    y_max_accel = max(np.max(np.abs(accel)), 1.2) * 1.2
    ax1.set_ylim(-y_max_accel, y_max_accel)

    # Gyroscope subplot
    ax2.plot(time, gyro[:, 0], color='green', alpha=0.7, label='Gyro X')
    ax2.plot(time, gyro[:, 1], color='purple', alpha=0.7, label='Gyro Y')

    ax2.scatter(time[gyro_x_encoded > 0], 1.4 * np.ones_like(time[gyro_x_encoded > 0]),
                color='green', marker='.', s=10, label='Gyro X +')
    ax2.scatter(time[gyro_x_encoded < 0], -1.4 * np.ones_like(time[gyro_x_encoded < 0]),
                color='green', marker='.', s=10, label='Gyro X -')

    ax2.scatter(time[gyro_y_encoded > 0], 1.5 * np.ones_like(time[gyro_y_encoded > 0]),
                color='purple', marker='.', s=10, label='Gyro Y +')
    ax2.scatter(time[gyro_y_encoded < 0], -1.5 * np.ones_like(time[gyro_y_encoded < 0]),
                color='purple', marker='.', s=10, label='Gyro Y -')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(10))


    ax2.set_title('Gyroscope Data')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (rad/s) / Spike')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Set y-axis limits for gyroscope
    y_max_gyro = max(np.max(np.abs(gyro)), 1.2) * 1.2
    ax2.set_ylim(-y_max_gyro, y_max_gyro)

    plt.tight_layout()
    plt.show()


