from typing import Tuple

import scipy as scp
import scipy.signal
import scipy.io.wavfile
import numpy as np
import matplotlib_tuda
import noisereduce as nr
import progressbar
from matplotlib import pyplot as plt, ticker
from matplotlib.contour import QuadContourSet

from src.plot_util import SubplotsAndSave

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
    :TODO: make it work with different stereo signals
    Reads Wave-Files 
    :param filename: path to file from media folder, without the .wav ending
    :returns: shape:(T,) all the sampled times as floats
    :returns: shape:(S,) the samples of the left stereoline as int16 values
    :returns: samplerate as float
    """
    rate, data = scipy.io.wavfile.read(f'../media/{filename}.wav')
    time_step = 1 / rate
    time_vec = np.arange(0, data.shape[0]) * time_step
    if len(data.shape) == 2:
        data = data[:, 0]
    return time_vec, data, rate


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


def plot_spectrogram(signal: np.ndarray, rate: float, caption: str, ax, show_xlabel: bool, show_ylabel: bool, show_title: bool) -> QuadContourSet:
    """
    Plots the spectrogram.
    :param signal: audio signal as int16 values
    :param rate: samplerate as float
    :param caption: The caption for the plot as String
    :param ax: Axes for subplot returning
    :param show_xlabel: whether to include an x label in subplot. Mostly only intended for last row.
    :param show_ylabel: whether to include an y label in subplot. Mostly only intended for first column.
    :param show_title: whether to include title in subplot. Mostly only intended for first row.
    :returns: subplot axes
    """
    freqs, times, spectrogram = scp.signal.spectrogram(signal, fs=rate)

    # prevent zero values
    spectrogram[spectrogram <= 0] = 1e-5

    c = ax.contourf(times, freqs, spectrogram, 10.0 ** np.arange(-6, 6), locator=ticker.LogLocator(), cmap=plt.get_cmap('jet'))
    if show_title:
        ax.set_title(f'Spectrogram ({caption})')
    if show_xlabel:
        ax.set_xlabel('Time window')
    if show_ylabel:
        ax.set_ylabel('Frequency band')
    # ax.set_yscale('log')
    plt.ylim(ymax=10000, ymin=0)
    return c


def main_plot_file(filenames, graphname: str, show_graph):
    """
    Generates required files (denoised, noise) and plots spectrograms.
    :param filenames: files to process as list. Files are rows.
    :param graphname: name of the saved plot file
    :param show_graph: list of plots and files to generate as booleans. [orig, denoised, noise]
    :return:
    """
    print('Generating plots for specified files. This may take a while.')
    plottypes = show_graph.count(True)
    with SubplotsAndSave('../res', graphname, nrows=len(filenames), ncols=plottypes + 1, sharey='all', file_types=['png']) as (fig, axs):
        bar = progressbar.ProgressBar(widgets=['Creating plot: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()],
                                      maxval=plottypes * len(filenames)).start()
        for i, filename in enumerate(filenames):
            time_vec, signal, rate = read_audio(filename)  # generate_exemplary_audio()

            if (show_graph[0]):
                # plot_signal(time_vec, signal, filename)
                c = plot_spectrogram(signal, rate, 'original', axs[i, 0], i == plottypes - 1, True, i == 0)
                bar.update(i * len(filenames) + 1)

            if (show_graph[0] | show_graph[1]):
                denoised_signal = denoise_audio(signal)

            if (show_graph[1]):
                scipy.io.wavfile.write(f'../media/{filename}_denoised_generated.wav', rate, denoised_signal)
                # plot_signal(time_vec, denoised_signal, filename + '_den')
                c = plot_spectrogram(denoised_signal, rate, 'denoised', axs[i, + show_graph[:1].count(True)], i == plottypes - 1, not show_graph[0], i == 0)
                bar.update(i * len(filenames) + show_graph[:1].count(True) + 1)

            if (show_graph[2]):
                noise_signal = signal - denoised_signal
                scipy.io.wavfile.write(f'../media/{filename}_noiseonly_generated.wav', rate, noise_signal)
                # plot_signal(time_vec, noise_signal, filename + '_noise')
                c = plot_spectrogram(noise_signal, rate, 'noise', axs[i, + show_graph[:2].count(True)], i == plottypes - 1, not (show_graph[0] | show_graph[1]), i == 0)
                bar.update(i * len(filenames) + show_graph[:2].count(True) + 1)

        bar.finish()
        print('Saving files to disk. Please stand by.')
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])  # left, bottom, width, height in percent
        fig.colorbar(c, cax=cbar_ax)
        fig.show()


def main_plot_place_comparision():
    print('Generating plots for places. This may take a while.')

    c = None
    # Recording places
    places = ['Fabi', 'Platz', 'Street', 'Tim', 'Treppe', 'Wald']
    # Recording ids
    ids = ['01', '02', '03', '04', '05']
    with SubplotsAndSave('../res', 'noiseenvs', nrows=len(ids), ncols=len(places) + 1, sharey='all', file_types=['png']) as (fig, axs):
        bar = progressbar.ProgressBar(widgets=['Creating plot: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()], maxval=len(places) * len(ids)).start()
        for i, end in enumerate(ids):
            for j, name in enumerate(places):
                filename = f'live/{name}{end}'
                time_vec, signal, rate = read_audio(filename)

                denoised_signal = denoise_audio(signal)

                noise_signal = signal - denoised_signal
                c = plot_spectrogram(noise_signal, rate, name + ', Noise', axs[i, j], i == len(ids) - 1, j == 0, i == 0)
                bar.update(i * len(places) + j + 1)
        bar.finish()
        print('Saving files to disk. Please stand by.')
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])  # left, bottom, width, height in percent
        fig.colorbar(c, cax=cbar_ax)
        fig.show()


def main():
    main_plot_place_comparision()
    main_plot_file(['tmp', 'tmp2'], 'plottest', [True, True, True])


if __name__ == '__main__':
    main()
