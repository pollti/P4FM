import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from multiband_spectral_substraction import segment


def environment_generator(*signals: np.ndarray, size: int = 256) -> np.ndarray:
    all_data = None
    frequencies = None
    for signal in signals:
        signal = segment(signal, size, 0.4, sp.signal.gaussian(size, size * 0.4))
        data = np.abs(np.fft.rfft(signal, axis=0))
        all_data = data if all_data is None else np.hstack((all_data, data))
    data = np.mean(all_data, axis=1)
    return data


def environment_detector(rate: int, signal: np.ndarray, *envs: np.ndarray, size: int = 256) -> np.ndarray:
    signal = segment(signal, size, 0.4, sp.signal.gaussian(size, size * 0.4))
    data = np.abs(np.fft.rfft(signal, axis=0))
    windows = data.shape[1]
    env_values = None
    for env in envs:
        errors = None
        for row in data.T:
            # any metric can be used here, scalar or ndarray. In case of ndarray: Caution with median later!!! Performance my be poor with arrays.
            tmp = squared_error_mean(row, env)
            errors = tmp if errors is None else np.hstack((errors, tmp))
        env_value = np.array([np.median(errors) / windows])
        env_values = env_value if env_values is None else np.hstack((env_values, env_value))
    return env_values


def squared_error_mean(a: np.ndarray, b: np.ndarray):
    return np.mean(squared_error(np.log(a), np.log(b)))


def squared_error(a: np.ndarray, b: np.ndarray):
    if a.shape != b.shape:
        print("Squared error: different dimensions – Crashing.")
    return np.square(a - b)


def plot_fourier(rate: int, *signals: np.ndarray, size: int = 256):
    all_data = None
    frequencies = np.fft.rfftfreq(size, 1 / rate)
    for signal in signals:
        signal = segment(signal, size, 0.4, sp.signal.gaussian(size, size * 0.4))
        data = np.fft.rfft(signal, axis=0)

        ### Only for the median plot:
        data = np.abs(data)  # maybe swap square and mean
        # data = np.sqrt(data ** 2)
        all_data = data if all_data is None else np.hstack((all_data, data))

        data = np.mean(data, 1)

        plt.scatter(frequencies, data, s=1)
    data = np.mean(all_data, axis=1)
    plt.scatter(frequencies, data, s=5)
    # for row in all_data.T:
    #    plt.scatter(frequencies, row, s=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((rate / size, rate / 2 + 1))
    plt.ylim((10 ** 1, 10 ** 5))
    # plt.ylim(ymin=0)
    plt.show()


def plot_fourier_mean(rate: int, *signals: np.ndarray):
    for signal in signals:
        data = np.fft.fft(signal)
        data = data ** 2
        for i in range(0, data.size - 10):
            data[i] = np.mean(data[i:i + 100])  # np.mean(data[i - 1:i])
        frequencies = np.fft.fftfreq(data.size, 1 / rate)
        plt.scatter(frequencies, data, s=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((10, rate / 2 + 1))
    plt.ylim((10 ** 0, 10 ** 16))
    # plt.ylim(ymin=0)
    plt.show()


def plot_fourier_pure(rate: int, *signals: np.ndarray):
    for signal in signals:
        data = np.fft.fft(signal)
        data = data ** 2
        frequencies = np.fft.fftfreq(data.size, 1 / rate)
        plt.scatter(frequencies, data, s=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((10, rate / 2 + 1))
    plt.ylim((10 ** 0, 10 ** 16))
    # plt.ylim(ymin=0)
    plt.show()
