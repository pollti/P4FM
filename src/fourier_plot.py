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
    data = np.median(all_data, axis=1)
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
            tmp = error_mean(row, env, True, False, False, False, rate, size)
            errors = tmp if errors is None else np.hstack((errors, tmp))
        env_value = np.array([np.median(errors)])
        env_values = env_value if env_values is None else np.hstack((env_values, env_value))
    return env_values


def error_mean(a: np.ndarray, b: np.ndarray, squared: bool, y_log: bool, x_log: bool, dga: bool, rate: int, size: int):
    if x_log and dga:
        print("Parameters incompatible: x_log and dga. Unexpected effects may occur.")
    if y_log:
        a = np.log(a)
        b = np.log(b)
    err = error(a, b, squared)
    if y_log: err = np.exp(err)  # necessary? Yes.
    weights = np.ones(a.size)
    if x_log:
        weights = weights / sp.fft.rfftfreq(size, 1 / rate)  # (np.arange(a.size) + np.ones(a.size))  # weights / np.exp(np.arange(a.size))
        weights[0] = 1
    if dga:
        freq = sp.fft.rfftfreq(size, 1 / rate)
        weights = sp.stats.norm.pdf(freq, 45, 20) + sp.stats.norm.pdf(freq, 1700, 4000)
        err = err * weights
    return np.sum(err) / np.sum(weights)  # if x_log else np.mean(err)


def error(a: np.ndarray, b: np.ndarray, squared: bool):
    if a.shape != b.shape:
        print("Squared error: different dimensions – Crashing.")
    return np.square(a - b) if squared else np.abs(a - b)


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
