from copy import deepcopy
from typing import Tuple

import sys

import scipy as scp
import scipy.signal
import scipy.io.wavfile
import numpy as np
import matplotlib_tuda
import noisereduce as nr
import progressbar
import webrtcvad
import fourier_plot
from matplotlib import pyplot as plt, ticker
from matplotlib.contour import QuadContourSet

from src.multiband_spectral_substraction import segment
from src.plot_util import SubplotsAndSave
from src.multiband_spectral_substraction import multiband_substraction_denoise

from sacred import Experiment

ex = Experiment()

matplotlib_tuda.load()
np.set_printoptions(linewidth=2000)


# noinspection PyUnusedLocal
@ex.config
def ex_config():
    de_signaled = False  # set true to use pre designaled recordings AND recordings_to_be_assigned
    environments_calculated = False  # set true to use precalculated environments

    path = "Audio/"  # The relative path to the recordings depends on working directory
    ending = ".wav"
    recordings = {"Room1": ["Fabi01", "Fabi02", "Fabi03", "Fabi04", "Fabi05"],
                  "Room2": ["Tim01", "Tim02", "Tim03", "Tim04", "Tim05"],
                  "Platz": ["Platz01", "Platz02", "Platz03", "Platz04", "Platz05"],
                  "Street": ["Street01", "Street02", "Street03", "Street04", "Street05"],
                  "Treppe": ["Treppe01", "Treppe02", "Treppe03", "Treppe04", "Treppe05"],
                  "Wald": ["Wald01", "Wald02", "Wald03", "Wald04", "Wald05"]}  # filenames assigned to location
    recordings_to_be_assigned = deepcopy(
        recordings)  # deepcopy notwendig um nicht nur die Referenz zu kopieren, for this default case we want to assign the already assigned data to see how acurate it works
    recordings_to_be_assigned["unknown"] = ["Fabi01"]  # put one file for test purpose
    environments = None
    recordings_noise_only = None
    recordings_to_be_assigned_noise_only = None
    # graphs
    image_folder = "res"
    show_graphs_denoising = [True, True, True]
    filename_graph_denoising = "file_denoising_steps"
    denoising_graph_multipass = True


def get_speech_postions(signal: np.ndarray, rate: float) -> Tuple[np.ndarray, int]:
    """
    Searches pure noise intervals. Somewhat configurable here.
    :param signal: signal to search in
    :param rate: sample rate of audio file
    :return: boolean whether frame contains speech (False -> pure noise)
    """
    # Integer sets filter aggressiveness. 0 classifies less as pure noise than 1 < 2 < 3. Change with vad.set_mode(1).
    vad = webrtcvad.Vad(0)
    samples_per_second = rate
    sample_rates = np.asarray([8000, 16000, 32000, 48000])
    if not (samples_per_second in sample_rates):
        new_sps = sample_rates[(np.abs(sample_rates - samples_per_second)).argmin()]
        # print(f' WARNING: Unexpected rate in noise position detection: {samples_per_second}. Falling back to {new_sps}')
        samples_per_second = new_sps
    samples_per_frame = samples_per_second // 100
    frame_count = signal.size // samples_per_frame
    splits = np.arange(samples_per_frame, samples_per_frame * (frame_count + 1), samples_per_frame, int)
    res = np.full(frame_count, False)
    for i, frame in enumerate(np.array_split(signal, splits)[:frame_count]):
        speech = vad.is_speech(frame.tobytes(), samples_per_second)
        if speech:
            res[i] = True
        #    print('Speech detected in segment.')
        # else:
        #    print('Only noise found in segment.')
    return res, samples_per_frame


def get_noise_intervals(signal: np.ndarray, rate: int) -> np.ndarray:
    """
    Returns intervals of pure noise in audio file.
    :param signal: the signal to search in
    :param rate: the samplerate
    :return: array of start and end points of intervals
    """
    speech, spf = get_speech_postions(signal, rate)
    res = []
    current_state = True
    start = 0
    for i, el in enumerate(speech):
        if current_state:
            if not el:
                current_state = False
                start = spf * i
        else:
            if el:
                current_state = True
                res.append([start, i * spf])
    return np.asarray(res)


def get_largest_noise_interval(intervals: np.ndarray) -> Tuple[int, int]:
    """"
    Returns largest of an array of intervals. Used to find pure noise interval.
    :param intervals: intervals as array
    :returns: starting point of largest interval
    :returns: end point of largest interval
    """
    largest_size = 0
    start = 0
    end = 0  # TODO: minimal noise interval must be specified here to avoid crashes!
    for a, b in intervals:
        if b - a > largest_size:
            start = a
            end = b
            largest_size = b - a
    return start, end


def denoise_audio(signal: np.ndarray, rate: int, including_multipass: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Denoises given audio signal.
    :param signal: audio signal as int16 values
    :param rate: audio sample rate in samples per second
    :param including_multipass: also use Multi-band Spectral subtraction for advanced denoising. Caution: returned audio length may then be smaller.
    :returns: denoised audio signal as int16 values
    :returns: intervals of pure noise
    :returns: start of used noise interval
    :returns: end of used noise interval
    """
    intervals = get_noise_intervals(signal, rate)
    a, b = get_largest_noise_interval(intervals)
    noisy_part = signal[a:b]
    # perform noise reduction
    reduced_noise = nr.reduce_noise(audio_clip=signal.astype(np.float16), noise_clip=noisy_part.astype(np.float16), verbose=False).astype(np.int16)
    noise_signal = signal - reduced_noise
    # Call multipass if needed
    if including_multipass:
        voice_leakage = multiband_substraction_denoise(noise_signal, rate, a, b)
        noise_signal = noise_signal[:voice_leakage.size] - voice_leakage
    return reduced_noise, noise_signal, intervals, a, b


def read_audio(path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Reads Wave-Files.
    :param path: path to file from media folder, without the .wav ending
    :returns: shape:(T,) all the sampled times as floats
    :returns: shape:(S,) the samples of the left stereoline as int16 values
    :returns: samplerate as float
    """
    rate, data = scipy.io.wavfile.read(f'{path}.wav')
    time_step = 1 / rate
    time_vec = np.arange(0, data.shape[0]) * time_step
    if len(data.shape) == 2:
        data = data[:, 0]
    return time_vec, data, int(rate)


def generate_exemplary_audio() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    :TODO: take params for signal functions for debugging purposes (currently hardcoded)
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
    plt.xlim(xmin=times[0], xmax=times[-1])
    plt.ylim(ymax=10000, ymin=0)
    return c


@ex.capture
def main_plot_file(filenames, filename_graph_denoising: str, show_graphs_denoising, denoising_graph_multipass: bool, path: str, image_folder: str):
    """
    Generates required files (denoised, noise) and plots spectrograms.
    :param filenames: files to process as list. Files are rows.
    :param filename_graph_denoising: name of the saved plot file
    :param show_graphs_denoising: list of plots and files to generate as booleans. [orig, denoised, noise]
    :param denoising_graph_multipass: Apply multipass approach for better denoising.
    :param path: path to audio files.
    :param image_folder: path to image files (for output).
    """
    print('Generating plots for specified files. This may take a while.')
    plottypes = show_graphs_denoising.count(True)
    if plottypes < 1:
        print("Warning: No plots selected for denoising information. Quitting plotting.")
        return
    # ncols + 1 for scala space. This generates an empty plot, but is the best solution for now.
    # CAUTION: if removed null pointers may occur later in axs[show_graph[:1].count(True)]
    with SubplotsAndSave(image_folder, filename_graph_denoising, nrows=len(filenames), ncols=plottypes + 1, sharey='all', file_types=['png']) as (fig, axs):
        bar = progressbar.ProgressBar(widgets=['Creating plot: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()],
                                      maxval=plottypes * len(filenames)).start()
        for i, filename in enumerate(filenames):
            time_vec, signal, rate = read_audio(path + filename)  # generate_exemplary_audio()
            #### START debugging shortener - ignore if not debugging. Used to reduce signal length for faster testing.
            # fact = 1
            # end = rate
            # signal = signal[::fact]
            # time_vec = time_vec[::fact]
            # rate = int(rate / fact)
            #### END debugging shortener

            # if only one file, axs is one-dimensional
            if axs.ndim > 1:
                axs_i = [axs[i, 0], axs[i, + show_graphs_denoising[:1].count(True)], axs[i, + show_graphs_denoising[:2].count(True)]]
            else:
                axs_i = [axs[0], axs[show_graphs_denoising[:1].count(True)], axs[show_graphs_denoising[:2].count(True)]]

            # Original audio including detected intervals for pure/ no noise
            if show_graphs_denoising[0]:
                # plot_signal(time_vec, signal, filename)
                ax = axs_i[0]
                c = plot_spectrogram(signal, rate, 'original', ax, i == plottypes - 1, True, i == 0)

                # Noise and speach lines at plot top.
                line_y = 9980

                intervals = get_noise_intervals(signal, rate)
                a, b = get_largest_noise_interval(intervals)

                # Colorize intervals of pe noise / speech.
                for k, (interval_start, interval_end) in enumerate(intervals):
                    # Assume no noise from beginning of time.
                    if k == 0:
                        ax.plot((time_vec[0], time_vec[interval_start]), (line_y, line_y), color='tuda:green', zorder=10)
                    # Assume no noise between intervals.
                    if k > 0:
                        ax.plot((time_vec[intervals[k - 1][1]], time_vec[interval_start]), (line_y, line_y), color='tuda:green', zorder=100)
                    # Assume no noise to end of time.
                    if k == intervals.shape[0] - 1:
                        ax.plot((time_vec[interval_end], time_vec[-1]), (line_y, line_y), color='tuda:green', zorder=10)
                    ax.plot((time_vec[interval_start], time_vec[interval_end]), (line_y, line_y), color='tuda:red', zorder=100)
                ax.plot((time_vec[b], time_vec[a]), (line_y, line_y), color='black', zorder=100)

                bar.update(i * plottypes + 1)

            # Generate denoised signal if needed.
            if show_graphs_denoising[1] or show_graphs_denoising[2]:
                denoised_signal, noise_signal, _, _, _ = denoise_audio(signal, rate, including_multipass=denoising_graph_multipass)

            # Denoised signal.
            if show_graphs_denoising[1]:
                scipy.io.wavfile.write(f'media/{filename}_denoised_generated.wav', rate, denoised_signal)
                # plot_signal(time_vec, denoised_signal, filename + '_den')
                c = plot_spectrogram(denoised_signal, rate, 'denoised', axs_i[1], i == plottypes - 1, not show_graphs_denoising[0], i == 0)
                bar.update(i * plottypes + show_graphs_denoising[:1].count(True) + 1)

            # Noise only.
            if show_graphs_denoising[2]:
                scipy.io.wavfile.write(f'media/{filename}_noiseonly_generated.wav', rate, noise_signal)
                # plot_signal(time_vec, noise_signal, filename + '_noise')
                c = plot_spectrogram(noise_signal, rate, 'noise', axs_i[2], i == plottypes - 1, not (show_graphs_denoising[0] | show_graphs_denoising[1]), i == 0)
                bar.update(i * plottypes + show_graphs_denoising[:2].count(True) + 1)

        bar.finish()
        print('Saving files to disk. Please stand by.')
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])  # left, bottom, width, height in percent
        fig.colorbar(c, cax=cbar_ax)
        fig.show()


def main_plot_place_comparision():  # TODO: crashes with only one file as parameter
    """
    Plots a comparision of denoised spectrums for environments with same number of files for each.
    """
    print('Generating plots for places. This may take a while.')

    c = None
    # Recording places
    places = ['Fabi', 'Platz', 'Street', 'Tim', 'Treppe', 'Wald']
    # Recording ids
    ids = ['01', '02', '03', '04', '05']
    with SubplotsAndSave('res', 'noiseenvs', nrows=len(ids), ncols=len(places) + 1, sharey='all', file_types=['png']) as (fig, axs):
        bar = progressbar.ProgressBar(widgets=['Creating plot: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()], maxval=len(places) * len(ids)).start()
        for i, end in enumerate(ids):
            for j, name in enumerate(places):
                filename = f'live/{name}{end}'
                time_vec, signal, rate = read_audio(filename)

                denoised_signal, noise_signal, intervals, a, b = denoise_audio(signal, rate)
                scipy.io.wavfile.write(f'media/{filename}_denoised_generated.wav', rate, denoised_signal)
                scipy.io.wavfile.write(f'media/{filename}_noiseonly_generated.wav', rate, noise_signal)

                ax = axs[i, j]
                c = plot_spectrogram(noise_signal, rate, name + ', Noise', ax, i == len(ids) - 1, j == 0, i == 0)
                # Noise and speach lines at plot top.
                line_y = 9980

                # Colorize intervals of pe noise / speech.
                for k, (interval_start, interval_end) in enumerate(intervals):
                    # Assume no noise from beginning of time.
                    if k == 0:
                        ax.plot((time_vec[0], time_vec[interval_start]), (line_y, line_y), color='tuda:green', zorder=10)
                    # Assume no noise between intervals.
                    if k > 0:
                        ax.plot((time_vec[intervals[k - 1][1]], time_vec[interval_start]), (line_y, line_y), color='tuda:green', zorder=100)
                    # Assume no noise to end of time.
                    if k == intervals.shape[0] - 1:
                        ax.plot((time_vec[interval_end], time_vec[-1]), (line_y, line_y), color='tuda:green', zorder=10)
                    ax.plot((time_vec[interval_start], time_vec[interval_end]), (line_y, line_y), color='tuda:red', zorder=100)

                ax.plot((time_vec[b], time_vec[a]), (line_y, line_y), color='black', zorder=100)
                bar.update(i * len(places) + j + 1)
        bar.finish()
        print('Saving files to disk. Please stand by.')
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])  # left, bottom, width, height in percent
        fig.colorbar(c, cax=cbar_ax)
        fig.show()


@ex.automain
def main(recordings, recordings_to_be_assigned, path, ending, de_signaled, environments_calculated, environments, recordings_noise_only, recordings_to_be_assigned_noise_only):
    # time_vec, signal, rate = read_audio('tmp')
    # segment(signal[:44100], 256)
    envs = {}  # Mapping from place to calculated environment
    if not de_signaled:
        recordings_noise_only = {}
        recordings_to_be_assigned_noise_only = {}
        for place, recs in recordings.items():
            signals = []
            designaled_signals = []
            for rec in recs:
                _, signal, rate = read_audio(path + rec)
                signals.append(signal)
                _, designaled_signal, _, _, _ = denoise_audio(signal, rate, False)
                designaled_signals.append(designaled_signal)

            envs[place] = fourier_plot.environment_generator(designaled_signals)
            recordings_noise_only[place] = designaled_signals

            # Optional: plot denoising spectrums
            main_plot_file(recs, filename_graph_denoising=f'file_denoising_steps_environment_{place}')

        for place, recs in recordings_to_be_assigned.items():
            signals = []
            designaled_signals = []
            for rec in recs:
                _, signal, rate = read_audio(path + rec)
                signals.append(signal)
                _, designaled_signal, _, _, _ = denoise_audio(signal, rate, False)
                designaled_signals.append(designaled_signal)

            recordings_to_be_assigned_noise_only[place] = designaled_signals

            # Optional: plot denoising spectrums
            main_plot_file(recs, filename_graph_denoising=f'file_denoising_steps_assigning_{place}')

    for place, denoised_signals in recordings_to_be_assigned_noise_only.items():
        print(f'Environment: {place}')
        for denoised_signal in denoised_signals:
            print(fourier_plot.environment_detector(rate, denoised_signal, *envs.values()))

    # Plot environments.
    frequencies = np.fft.rfftfreq(256, 1 / rate)
    for place, env in envs.items():
        plt.scatter(frequencies, env, s=1, label=place)
    # for row in all_data.T:
    #    plt.scatter(frequencies, row, s=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((rate / 256, rate / 2 + 1))
    plt.ylim((10 ** 1, 10 ** 5))
    plt.xlabel("frequency")
    plt.ylabel("energy")
    plt.title("Environment data")
    plt.show()
