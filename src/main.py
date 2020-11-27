from typing import Tuple

import scipy as scp
import scipy.signal
import scipy.io.wavfile
import numpy as np
import matplotlib_tuda
import noisereduce as nr
from matplotlib import pyplot as plt, ticker

matplotlib_tuda.load()


def denoise_audio(signal: np.ndarray) -> np.ndarray:
    """
    :TODO: recognize the noise to automate the denoising.
    Denoises given audio signal.
    :param signal: audio signal as int16 values
    :returns: denoised audio signal as int16 values
    """
    noisy_part = signal[:3 * 44100]
    # perform noise reduction
    reduced_noise = nr.reduce_noise(audio_clip=signal.astype(np.float16), noise_clip=noisy_part.astype(np.float16), verbose=False).astype(np.int16)
    return reduced_noise


def read_audio(filename: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    :TODO: make it work with stereo/mono
    Reads Wave-Files 
    :param filename: path to file from media folder, without the .wav ending
    :returns: shape:(T,) all the sampled times as floats
    :returns: shape:(S,) the samples of the left stereoline as int16 values
    :returns: samplerate as float
    """
    rate, data = scipy.io.wavfile.read(f'../media/{filename}.wav')
    time_step = 1 / rate
    time_vec = np.arange(0, data.shape[0]) * time_step
    return time_vec, data[:, 0], rate


def generate_exemplary_audio() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    :TODO: make it work with stereo/mono
    :TODO: take params for signalfunctions
    Generates exemplary audio signal
    :returns: shape:(T,) all the sampled times as floats
    :returns: shape:(S,) the samples of the left stereoline as int16 values
    :returns: samplerate as float
    """
    time_step = .01
    time_vec = np.arange(0, 70, time_step)
    signal = np.sin(0.5 * np.pi * time_vec * (1 + .1 * time_vec))
    return time_vec, signal, 1 / time_step


def plot_signal(time_vec: np.ndarray, signal: np.ndarray, caption: str) -> None:
    """
    plots the audio signal
    :param time_vec: all the sampled times as floats
    :param signal: audio signal as int16 values
    :param caption: The caption for the plot as String
    """
    fig, ax = plt.subplots()
    ax.set_title(f'Signal ({caption})')
    ax.plot(time_vec, signal)
    fig.show()


def plot_spectrogram(signal: np.ndarray, rate: float, caption: str) -> None:
    """
    Plots the spectogram.
    :param signal: audio signal as int16 values
    :param rate: samplerate as float
    :param caption: The caption for the plot as String
    """
    freqs, times, spectrogram = scp.signal.spectrogram(signal, fs=rate)

    fig, ax = plt.subplots()
    tmp = ax.contourf(times, freqs, spectrogram, 10.0 ** np.arange(-6, 6), locator=ticker.LogLocator(), cmap=plt.get_cmap('jet'))
    # plt.clim(10^(-12),10)
    fig.colorbar(tmp)
    ax.set_title(f'Spectrogram ({caption})')
    ax.set_ylabel('Frequency band')
    # ax.set_yscale('log')
    plt.ylim(ymax=10000, ymin=0)
    ax.set_xlabel('Time window')
    fig.show()


def main():
    # Generate plots for different file stadiums
    print('Generating plots. This may take a while.')

    for ending in ['original']:
        filename = f'test01_{ending}'
        time_vec, signal, rate = read_audio(filename)

        denoised_signal = denoise_audio(signal)
        scipy.io.wavfile.write(f'../media/{filename}_denoised_generated.wav', rate, denoised_signal)
        plot_signal(time_vec, denoised_signal, filename + '_den')
        plot_spectrogram(denoised_signal, rate, filename + '_den')

        noise_signal = signal - denoised_signal
        scipy.io.wavfile.write(f'../media/{filename}_noiseonly_generated.wav', rate, noise_signal)
        plot_signal(time_vec, noise_signal, filename + '_noise')
        plot_spectrogram(noise_signal, rate, filename + '_noise')

    for ending in ['original', 'noise', 'ton']:
        filename = f'test01_{ending}'
        time_vec, signal, rate = read_audio(filename)  # generate_exemplary_audio()
        # plot_signal(time_vec, signal, filename)
        plot_spectrogram(signal, rate, filename)
    
    input('Press enter to finish.')


if __name__ == '__main__':
    main()
