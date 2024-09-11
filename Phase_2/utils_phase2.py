import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import numpy as np
import os


def _calculate_and_plot_from_Q(gt_t_xyz, gt_q_wxyz, plot_data=0):
    """
    Plot the trajectory of a dataset in the world orientation and return translated points
    :param gt_t_xyz: positions of the body in world coordinates
    :param gt_q_wxyz: heading of the body in world coordinates
    :return: array of translated points and plot in world coordinates
    """
    tx, ty, tz = gt_t_xyz[:, 0], gt_t_xyz[:, 1], gt_t_xyz[:, 2]
    qw, qx, qy, qz, = gt_q_wxyz[:, 0], gt_q_wxyz[:, 1], gt_q_wxyz[:, 2], gt_q_wxyz[:, 3]
    # Convert quaternions to rotation objects
    rotations = Rotation.from_quat(np.column_stack((qx, qy, qz, qw)))

    # Create a single point for the drone (center of mass)
    drone_point = np.array([0, 0, 0])

    # Initialize array to store all translated points
    all_translated_points = np.zeros_like(gt_t_xyz)

    # Calculate all translated points
    for i in range(len(gt_t_xyz)):
        rotated_point = rotations[i].apply(drone_point)
        all_translated_points[i] = rotated_point + np.array([tx[i], ty[i], tz[i]])

    if plot_data:
        # Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory of translated points
        ax.plot(all_translated_points[:, 0], all_translated_points[:, 1], all_translated_points[:, 2], 'b-',
                linewidth=2, label='Trajectory')

        # Plot drone positions at intervals
        N = gt_t_xyz.shape[0]
        interval = N // 20  # Adjust this to change the number of positions shown
        for i in range(0, N, interval):
            ax.plot([all_translated_points[i, 0]], [all_translated_points[i, 1]], [all_translated_points[i, 2]], 'ro')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Drone Trajectory and Positions')

        plt.tight_layout()
        plt.show()

    return all_translated_points


def read_imu_data(dir_path):
    """
    Read IMU data from a text file
    :param txt_path:
    :return: An array with the IMU data in this order ts, wx,wy,wz, ax,ay,az
    Args:
        dir_path: :
    """
    serial_num = []
    timestamps = []
    wx_values = []
    wy_values = []
    wz_values = []
    ax_values = []
    ay_values = []
    az_values = []
    txt_path = os.path.join(dir_path, 'IMU.txt')
    with open(txt_path, mode='r') as imu_file:
        next(imu_file)  # Skip the first line
        for line in imu_file:
            line = line.strip().split()
            sn, ts, wx, wy, wz, ax, ay, az = map(float, line)
            serial_num.append(sn)
            timestamps.append(ts)
            wx_values.append(wx)
            wy_values.append(wy)
            wz_values.append(wz)
            ax_values.append(ax)
            ay_values.append(ay)
            az_values.append(az)

    imu_data = np.zeros((len(ax_values), 7))
    imu_data[:, 0] = timestamps
    imu_data[:, 1] = wx_values
    imu_data[:, 2] = wy_values
    imu_data[:, 3] = wz_values
    imu_data[:, 4] = ax_values
    imu_data[:, 5] = ay_values
    imu_data[:, 6] = az_values
    return imu_data


def breakdown_data(data_path, start=0, stop=-1, plot_data=0, fix_axis_and_g=0, no_Z_gyro=0, no_Z_accel=0, fix_meas=0,
                   g=9.8065):
    """
    Function that breaks down the data into parts
    :param data_path: path to the IMU_data.txt file
    :param start: starting point of dataset (in order to remove calibration)
    :param stop: end point of dataset
    :param plot_data: whether to plot the raw data
    :param fix_axis_and_g: if Y is -9.85, the IMU is set on its side, so rotate it
    :param no_Z_gyro: return the IMU data without the Z axis in gyro
    :param no_Z_accel: return the IMU data without the Z axis accel
    :param fix_meas: whether to fix measurements
    :param g: gravity constant
    :param make_as_unsigned: shift all accel and gyro data up by this value (default 5)
    :return: ts, gyro, accel
    """
    data = read_imu_data(data_path)
    ts = data[start:stop, 0]
    # UZH Data set default: rad/s, use np.degrees(  ) in order to convert to 1/s
    gyro = data[start:stop, 1:4]
    # UZH Data set default: m/(s^2), use ()/(9.8065) in order to convert to g
    accel = data[start:stop, 4:7]
    ts = ts - ts[0]

    if fix_meas:
        accel, gyro = fix_measurements_and_g(accel, gyro)  # for datasets in deg/s and g + decrease g

    if fix_axis_and_g:
        accel, gyro = fix_axis_an_g(accel, gyro, g)

    if plot_data:
        plot_raw_data(ts, gyro, accel)

    if no_Z_gyro:
        gyro = gyro[:, 0:2]

    if no_Z_accel:
        accel = accel[:, 0:2]

    return ts, gyro, accel


def fix_axis_an_g(accel, gyro, g):
    """
    Rotate if Y is pointed to Z, and reduce the g from Z axis
    :param accel:
    :param gyro:
    :param g:
    :return:
    """
    gravity_vec = np.array([0, 0, -g])
    R_y_to_z = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    # fix axis
    return (np.dot(accel, R_y_to_z.T) - gravity_vec), (np.dot(gyro, R_y_to_z.T))


def fix_measurements_and_g(accel, gyro):
    """
    Return the data in m/(s^2) and rad/s
    :param accel:
    :param gyro:
    :return:
    """
    accel = (accel * 9.8065)
    accel[:, 2] -= 9.8065
    return accel, (gyro * (np.pi / 180))


def get_gt_data(dir_path,plot_data=0):
    """
    load data from Groundtruth file
    :param dir_path: path to directory of the GT file
    :return: t_xyz, q_wxyz
    """
    gt_path = os.path.join(dir_path, 'groundtruth.txt')
    gt_data = np.loadtxt(gt_path, skiprows=1)
    gt_ts = gt_data[:, 0]
    gt_t_xyz = gt_data[:, 1:4]
    gt_q_wxyz = gt_data[:, 4:]

    translated_points = _calculate_and_plot_from_Q(gt_t_xyz, gt_q_wxyz, plot_data=plot_data)
    return gt_ts, gt_t_xyz, gt_q_wxyz, gt_data, translated_points


def plot_raw_data(cal_timestamp, cal_gyroscope, cal_accelerometer):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle(' IMU Data Visualization', fontsize=16)

    # Gyroscope data (left column)
    axs[0, 0].plot(cal_timestamp, cal_gyroscope[:, 0])
    axs[0, 0].set_title('Gyro X')
    axs[0, 0].set_ylabel('Angular velocity (rad/s)')

    axs[1, 0].plot(cal_timestamp, cal_gyroscope[:, 1])
    axs[1, 0].set_title('Gyro Y')
    axs[1, 0].set_ylabel('Angular velocity (rad/s)')

    axs[2, 0].plot(cal_timestamp, cal_gyroscope[:, 2])
    axs[2, 0].set_title('Gyro Z')
    axs[2, 0].set_ylabel('Angular velocity (rad/s)')
    axs[2, 0].set_xlabel('Time (s)')

    # Accelerometer data (right column)
    axs[0, 1].plot(cal_timestamp, cal_accelerometer[:, 0])
    axs[0, 1].set_title('Accel X')
    axs[0, 1].set_ylabel('Acceleration (m/s²)')

    axs[1, 1].plot(cal_timestamp, cal_accelerometer[:, 1])
    axs[1, 1].set_title('Accel Y')
    axs[1, 1].set_ylabel('Acceleration (m/s²)')

    axs[2, 1].plot(cal_timestamp, cal_accelerometer[:, 2])
    axs[2, 1].set_title('Accel Z')
    axs[2, 1].set_ylabel('Acceleration (m/s²)')
    axs[2, 1].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()


def generate_and_plot_parameter_arrays(plot_param=0, grid_size=50, theta=-0.45, lp_range=(10, 15),
                                       lf=1, w_amplifier=3):
    center = grid_size // 2
    y, x = np.ogrid[-center:center, -center:center]
    distances = np.sqrt(x ** 2 + y ** 2) / (center * np.sqrt(2))

    angles = np.arctan2(y, x)
    weights_X = w_amplifier * np.cos(-angles)
    weights_Y = w_amplifier * np.sin(-angles)

    LP_array = lp_range[0] + (lp_range[1] - lp_range[0]) * np.exp(np.minimum(distances, 1) - 1)

    if plot_param:
        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(24, 8))  # 1 row, 3 columns
        fig.suptitle('Grid Parameters Visualization', fontsize=24, y=1.05)

        im1 = axs[0].imshow(weights_X, cmap='coolwarm')
        axs[0].set_title('Weights X', fontsize=18, pad=20)
        fig.colorbar(im1, ax=axs[0], shrink=0.8)

        im2 = axs[1].imshow(weights_Y, cmap='coolwarm')
        axs[1].set_title('Weights Y', fontsize=18, pad=20)
        fig.colorbar(im2, ax=axs[1], shrink=0.8)

        im3 = axs[2].imshow(LP_array, cmap='viridis')
        axs[2].set_title('Leakage Period', fontsize=18, pad=20)
        fig.colorbar(im3, ax=axs[2], shrink=0.8)

        plt.tight_layout()
        plt.show()

    return weights_X, weights_Y, LP_array, lf, theta
