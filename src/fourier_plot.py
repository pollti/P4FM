import numpy as np
import matplotlib.pyplot as plt


def plot_fourier(rate: int, *signals: np.ndarray):
    for signal in signals:
        data = np.fft.fft(signal)
        data = data ** 2
        frequencies = np.fft.fftfreq(signal.size, 1 / rate)
        plt.scatter(frequencies, data, s=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((10, rate / 2 + 1))
    plt.ylim((10 ** 0, 10 ** 16))
    # plt.ylim(ymin=0)
    plt.show()
