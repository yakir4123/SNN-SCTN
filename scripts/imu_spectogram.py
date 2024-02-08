import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
file_path = '../dataset/RawAccel.csv'  # Update with your file path
df = pd.read_csv(file_path)

# Extract the accelerometer data (x, y, z) from the DataFrame
# Assuming the columns are labeled as 'accel_x', 'accel_y', and 'accel_z'
acceleration_data = df[[' x', ' y', ' z']].values

# Compute the time differences between consecutive timestamps
timestamps = df['Timpstemp'].values
time_intervals = np.diff(timestamps)

# Compute the sampling frequency
sampling_freq = 1 / np.mean(time_intervals)

# Perform FFT on the acceleration data
fft_output = np.fft.fft(acceleration_data, axis=0)
freqs = np.fft.fftfreq(len(acceleration_data), d=1/sampling_freq)

# Only consider positive frequencies
positive_freqs = freqs[:len(freqs)//2]
positive_fft_output = np.abs(fft_output[:len(freqs)//2])

# Plot the frequency spectrum for each axis
plt.figure(figsize=(10, 6))
plt.plot(positive_freqs, positive_fft_output)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum (Positive Frequencies)')
plt.legend(['x-axis', 'y-axis', 'z-axis'])
plt.show()
