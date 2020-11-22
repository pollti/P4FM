from typing import Tuple

import scipy as scp
import scipy.signal
import scipy.io.wavfile
import numpy as np
from matplotlib import pyplot as plt, ticker, cm


def read_audio(filename: str) -> tuple[np.ndarray, np.ndarray, float]:
    rate, data = scipy.io.wavfile.read(f'../media/{filename}.wav')
    time_step = 1 / rate
    time_vec = np.arange(0, data.shape[0]) * time_step
    return time_vec, data[:, 0], rate

    # Seed the random number generator
    np.random.seed(0)

    time_step = .01
    time_vec = np.arange(0, 70, time_step)

    # A signal with a small frequency chirp
    signal = np.sin(0.5 * np.pi * time_vec * (1 + .1 * time_vec))
    return time_vec, signal


def plot_signal(time_vec: np.ndarray, signal: np.ndarray, filename: str) -> None:
    fig, ax = plt.subplots()
    ax.set_title(f'Signal ({filename})')
    ax.plot(time_vec, signal)
    fig.show()


def plot_spectrogram(signal: np.ndarray, frequency: float, filename: str) -> None:
    freqs, times, spectrogram = scp.signal.spectrogram(signal, fs=frequency)

    fig, ax = plt.subplots()
    tmp = ax.contourf(times, freqs, spectrogram, 10.0 ** np.arange(-6, 6), locator=ticker.LogLocator(), cmap=plt.get_cmap('jet'))
    # plt.clim(10^(-12),10)
    fig.colorbar(tmp)
    ax.set_title(f'Spectrogram ({filename})')
    ax.set_ylabel('Frequency band')
    # ax.set_yscale('log')
    plt.ylim(ymax=10000, ymin=0)
    ax.set_xlabel('Time window')
    fig.show()


def main():
    print('Hello, world!')
    for i in ['original', 'noise', 'ton']:
        filename = f'test01_{i}'
        time_vec, signal, frequency = read_audio(filename)
        plot_signal(time_vec, signal, filename)
        plot_spectrogram(signal, frequency, filename)


if __name__ == '__main__':
    main()
