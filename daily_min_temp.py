# Modelling daily minimum temperature in Brampton, Ontario (43.75 N, 79.75W)- ID = 8557.
import os

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_data(time, data, start=0, end=None, format="-"):
    plt.plot(time[start:end], data[start:end], format),
    plt.xlabel("Time (day)")
    plt.ylabel("Temperature (C)")


mat = scipy.io.loadmat('tasmin_ECMWF_8964X365X38.mat')
# Data is provided as int16 with a scale factor of 10. Divide by 10.0 before use.
# See http://lamps.math.yorku.ca/OntarioClimate/index_app_data.htm#/historicaldata

mat_scaled = mat['outputData'] / 10.0
# Data shape: (grid point, day[1:365], year[1979:2016])

data = mat_scaled[8557].flatten()  # Extract data for Brampton, Ontario grid point (ID = 8557) and flatten

time = np.arange(1, 38 * 365 + 1, 1)  # Time in days from Day 1 to 365 for 38 years

# Using fft (Fast Fourier Transform), we can attain periodicity information that can de-trend the data
# This is a sanity check to make sure a frequency of 1 is returned (temperature data should only repeat yearly)
fft = tf.signal.rfft(data)
freq_dataset = np.arange(0, len(fft)).astype(np.float)

freq_dataset[0] = 0.2  # To handle 0 in a log scale

plt.step(freq_dataset, np.abs(fft))
plt.xscale('log')
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1], labels=['1/Year'])

plt.xlabel('Frequency (logarithmic)')
plt.show()

# Deseasonalizing data
fund_freq = np.argmax(fft)
amplitude = np.asarray(fft)[1].imag / (2 * len(data))  # Extract sine wave contribution of 1/Year from fft
yearly_adjust = amplitude * np.sin(time * (2 * np.pi * fund_freq / 365)) + np.mean(data)

adjusted_data = data - yearly_adjust

# Can alternatively fit polynomial: x^2*b1 + x*b2 + ... + bn to avoid fft

# X = [i % 365 for i in range(0, len(data))]
# degree = 4
# coef = np.polyfit(X, data, degree)
# print('Coefficients: %s' % coef)
# # create curve
# curve = list()
# for i in range(len(X)):
#     value = coef[-1]
#     for d in range(degree):
#         value += X[i] ** (degree - d) * coef[d]
#     curve.append(value)
# # create seasonally adjusted
# values = data
# diff = list()
# for i in range(len(data)):
#     value = data[i] - curve[i]
#     diff.append(value)
# plt.plot(diff)
# plt.show()


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)  # creates an extra dimension so sampling can be done
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)  # drop_remainder allows for uniform sized datasets
    # do not forget to shift the window (default is 0)!
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))  # flattens the tensor and batches it again into
    # desired window size. win_size + 1 is used to include the "+1" from the y value
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1:]))  # takes windows and splits into tuple of (x_value, y_value)
    return ds.batch(batch_size).prefetch(1)


batch_size = 256
window_size = 365
shuffle_buffer = 1000
split_ind = int(0.7 * len(data))

train_ds = windowed_dataset(adjusted_data, window_size, batch_size, shuffle_buffer)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(365, kernel_size=5, activation='relu', input_shape=[None, 1], padding='causal'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=2e-4, momentum=0.90), loss=tf.keras.losses.Huber(), metrics=['mae'])
model.summary()

num_epochs = 150
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))

# save_model = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, save_best_only=True)

history = model.fit(train_ds, epochs=num_epochs)

# plt.semilogx(history.history['lr'], history.history['mae'])
plt.plot(range(len(history.history['mae'])), history.history['mae'])
plt.show()

model.save('min_temp_model2.h5')
