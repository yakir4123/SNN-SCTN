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
    accel (numpy.ndarray): Acceleration data with shape (n, 3)
    gyro (numpy.ndarray): Gyroscope data with shape (n, 3)

    Returns:
    numpy.ndarray: 2D array with shape (n, 3) where each column is an encoded signal
                   (accel_x, accel_y, gyro_z) and each row is a timestamp
    """
    # Ensure we're using x and y components of accel, and z component of gyro
    accel = accel[:, :2]
    gyro_z = gyro[:, 2]

    # Encode data
    accel_x_encoded = encode_and_combine(accel[:, 0])
    accel_y_encoded = encode_and_combine(accel[:, 1])
    gyro_z_encoded = encode_and_combine(gyro_z)

    # Combine all encoded data into a single 2D array
    combined_data = np.column_stack([accel_x_encoded, accel_y_encoded, gyro_z_encoded])
    if plot_data:
        _plot_poisson_encoded_imu_data(accel, gyro_z, accel_x_encoded, accel_y_encoded, gyro_z_encoded)

    if split:
        return np.column_stack([accel_x_encoded, accel_y_encoded]), gyro_z_encoded.reshape(-1, 1)

    return combined_data



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



def _plot_poisson_encoded_imu_data(accel, gyro_z, accel_x_encoded, accel_y_encoded, gyro_z_encoded,
                                   time_window=0.01, max_rate=100, threshold=0.2):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))  # Reduced figure size

    time = np.arange(len(accel)) * time_window

    # Accelerometer subplot
    ax1.plot(time, accel[:, 0], color='blue', alpha=0.7, label='Accel X')
    ax1.plot(time, accel[:, 1], color='red', alpha=0.7, label='Accel Y')

    ax1.scatter(time[accel_x_encoded > 0], 5 * np.ones_like(time[accel_x_encoded > 0]),
                color='blue', marker='.', s=5, label='Accel X +')  # Reduced marker size
    ax1.scatter(time[accel_x_encoded < 0], -5 * np.ones_like(time[accel_x_encoded < 0]),
                color='blue', marker='.', s=5, label='Accel X -')

    ax1.scatter(time[accel_y_encoded > 0], 5.2 * np.ones_like(time[accel_y_encoded > 0]),
                color='red', marker='.', s=5, label='Accel Y +')
    ax1.scatter(time[accel_y_encoded < 0], -5.2 * np.ones_like(time[accel_y_encoded < 0]),
                color='red', marker='.', s=5, label='Accel Y -')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    ax1.set_title('Accelerometer Data', fontsize=12)  # Adjusted font size
    ax1.set_ylabel('Acceleration (m/s²) / Spike', fontsize=10)
    ax1.legend(loc='upper right', fontsize=8)  # Moved legend and adjusted font size
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(50))  # Adjusted tick frequency

    # Set y-axis limits for accelerometer
    y_max_accel = max(np.max(np.abs(accel)), 1.2) * 1.2
    ax1.set_ylim(-y_max_accel, y_max_accel)

    # Gyroscope subplot (Z-axis only)
    ax2.plot(time, gyro_z, color='green', alpha=0.7, label='Gyro Z')

    ax2.scatter(time[gyro_z_encoded > 0], 1.4 * np.ones_like(time[gyro_z_encoded > 0]),
                color='green', marker='.', s=5, label='Gyro Z +')
    ax2.scatter(time[gyro_z_encoded < 0], -1.4 * np.ones_like(time[gyro_z_encoded < 0]),
                color='green', marker='.', s=5, label='Gyro Z -')

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(50))  # Adjusted tick frequency

    ax2.set_title('Gyroscope Data (Z-axis)', fontsize=12)  # Adjusted font size
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('Angular Velocity (rad/s) / Spike', fontsize=10)
    ax2.legend(loc='upper right', fontsize=8)  # Moved legend and adjusted font size
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Set y-axis limits for gyroscope
    y_max_gyro = max(np.max(np.abs(gyro_z)), 1.2) * 1.2
    ax2.set_ylim(-y_max_gyro, y_max_gyro)

    plt.tight_layout(pad=2.0)  # Increased padding
    plt.show()


