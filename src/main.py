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
from src.multiband_spectral_substraction import SSMultibandKamath02

matplotlib_tuda.load()


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
    if including_multipass:
        voice_leakage = SSMultibandKamath02(noise_signal, rate, a, b)
        noise_signal = noise_signal[:voice_leakage.size] - voice_leakage
    return reduced_noise, noise_signal, intervals, a, b


def read_audio(filename: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    :TODO: make it work with different stereo signals
    Reads Wave-Files 
    :param filename: path to file from media folder, without the .wav ending
    :returns: shape:(T,) all the sampled times as floats
    :returns: shape:(S,) the samples of the left stereoline as int16 values
    :returns: samplerate as float
    """
    rate, data = scipy.io.wavfile.read(f'media/{filename}.wav')
    time_step = 1 / rate
    time_vec = np.arange(0, data.shape[0]) * time_step
    if len(data.shape) == 2:
        data = data[:, 0]
    return time_vec, data, int(rate)


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
    plt.xlim(xmin=times[0], xmax=times[-1])
    plt.ylim(ymax=10000, ymin=0)
    return c


def main_plot_file(filenames, graphname: str, show_graph):  # TODO: crashes with only one file as parameter
    """
    Generates required files (denoised, noise) and plots spectrograms.
    :param filenames: files to process as list. Files are rows.
    :param graphname: name of the saved plot file
    :param show_graph: list of plots and files to generate as booleans. [orig, denoised, noise]
    :return:
    """
    print('Generating plots for specified files. This may take a while.')
    plottypes = show_graph.count(True)
    with SubplotsAndSave('res', graphname, nrows=len(filenames), ncols=plottypes + 1, sharey='all', file_types=['png']) as (fig, axs):
        bar = progressbar.ProgressBar(widgets=['Creating plot: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()],
                                      maxval=plottypes * len(filenames)).start()
        for i, filename in enumerate(filenames):
            time_vec, signal, rate = read_audio(filename)  # generate_exemplary_audio()
            #### START debugging shortener
            fact = 1
            end = rate
            signal = signal[::fact]
            time_vec = time_vec[::fact]
            rate = int(rate / fact)
            #### END debugging shortener

            if show_graph[0]:
                # plot_signal(time_vec, signal, filename)
                ax = axs[i, 0]
                c = plot_spectrogram(signal, rate, 'original', ax, i == plottypes - 1, True, i == 0)

                # Noise and speach lines at plot top.
                line_y = 9980

                intervals = get_noise_intervals(signal, rate)
                a, b = get_largest_noise_interval(intervals)

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
                # verticalines if needed
                # for interval in intervals:
                #    ax.axvline(interval[0] / rate, color='tuda:green', ls='dotted')
                #    ax.axvline(interval[1] / rate, color='tuda:red', ls='dashed')

                bar.update(i * plottypes + 1)

            if show_graph[0] or show_graph[1]:
                denoised_signal, noise_signal, _, _, _ = denoise_audio(signal, rate)

            if show_graph[1]:
                scipy.io.wavfile.write(f'media/{filename}_denoised_generated.wav', rate, denoised_signal)
                # plot_signal(time_vec, denoised_signal, filename + '_den')
                c = plot_spectrogram(denoised_signal, rate, 'denoised', axs[i, + show_graph[:1].count(True)], i == plottypes - 1, not show_graph[0], i == 0)
                bar.update(i * plottypes + show_graph[:1].count(True) + 1)

            if show_graph[2]:
                scipy.io.wavfile.write(f'media/{filename}_noiseonly_generated.wav', rate, noise_signal)
                # plot_signal(time_vec, noise_signal, filename + '_noise')
                c = plot_spectrogram(noise_signal, rate, 'noise', axs[i, + show_graph[:2].count(True)], i == plottypes - 1, not (show_graph[0] | show_graph[1]), i == 0)
                bar.update(i * plottypes + show_graph[:2].count(True) + 1)

        bar.finish()
        print('Saving files to disk. Please stand by.')
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])  # left, bottom, width, height in percent
        fig.colorbar(c, cax=cbar_ax)
        fig.show()


def main_plot_place_comparision():  # TODO: crashes with only one file as parameter
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
                # verticalines if needed
                # for interval in intervals:
                #    ax.axvline(interval[0] / rate, color='tuda:green', ls='dotted')
                #    ax.axvline(interval[1] / rate, color='tuda:red', ls='dashed')
                bar.update(i * len(places) + j + 1)
        bar.finish()
        print('Saving files to disk. Please stand by.')
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])  # left, bottom, width, height in percent
        fig.colorbar(c, cax=cbar_ax)
        fig.show()


def main():
    time_vec, signal, rate = read_audio('tmp')
    segment(signal[:44100], 256)

    if len(sys.argv) < 2:
        eingabe = input('Please give a filename or nothing to terminate: Press enter to finish.')
        eingabeListe = eingabe.split(" ")
        if not eingabeListe:  # ergibt True falls Liste keine Einträge hat
            quit(1)  # TODO: Returncodes & fix list with empty string
    else:
        eingabeListe = sys.argv[1:]

    for name in eingabeListe:
        print(name)
    # main_plot_place_comparision()
    # main_plot_file(['DEMO_multipass'], 'plottest', [True, True, True])

    _, signal1, rate = read_audio(f'noisewhite')
    _, signal2, _ = read_audio(f'noisepink')
    _, signal3, _ = read_audio(f'noisebrownian')
    _, signal4, _ = read_audio(f'biiiiiiep')
    # fourier_plot.plot_fourier(rate, signal1, signal2, signal3, signal4)

    places = ['Fabi', 'Platz', 'Street', 'Tim', 'Treppe', 'Wald']
    # places = ['Platz', 'Street', 'Wald']
    envs = []
    d = {}
    for place in places:
        _, signal1, rate = read_audio(f'live/{place}01')
        _, signal2, _ = read_audio(f'live/{place}02')
        _, signal3, _ = read_audio(f'live/{place}03')
        _, signal4, _ = read_audio(f'live/{place}04')
        _, signal5, _ = read_audio(f'live/{place}05')
        # _, signal6, _ = read_audio(f'live/{place}06')
        # _, signal7, _ = read_audio(f'live/{place}07')
        _, signal1n, _, _, _ = denoise_audio(signal1, rate, False)
        _, signal2n, _, _, _ = denoise_audio(signal2, rate, False)
        _, signal3n, _, _, _ = denoise_audio(signal3, rate, False)
        _, signal4n, _, _, _ = denoise_audio(signal4, rate, False)
        _, signal5n, _, _, _ = denoise_audio(signal5, rate, False)
        # _, signal6n, _, _, _ = denoise_audio(signal6, rate, False)
        # _, signal7n, _, _, _ = denoise_audio(signal7, rate, False)
        # plot_fourier(rate, signal1n, signal2n, signal3n, signal4n, signal5n, signal6n, signal7n)
        # fourier_plot.plot_fourier(rate, signal1n, signal2n, signal3n, signal4n, signal5n)
        envs.append(fourier_plot.environment_generator(signal1n, signal2n, signal3n, signal4n, signal5n))
        d[place] = [signal1n, signal2n, signal3n, signal4n, signal5n]

    np.set_printoptions(linewidth=2000)
    #for place in places:
    #    print(f'Environment: {place}')
    #    for signal in d[place]:
    #        print(fourier_plot.environment_detector(rate, signal, *envs, size=1024))
    places = ['nDecke', 'nRaum', 'nFenster']
    for place in places:
        print(f'Environment: {place}')
        _, signal1, rate = read_audio(f'live/{place}01')
        _, signal2, _ = read_audio(f'live/{place}02')
        _, signal1n, _, _, _ = denoise_audio(signal1, rate, False)
        _, signal2n, _, _, _ = denoise_audio(signal2, rate, False)
        print(fourier_plot.environment_detector(rate, signal1n, *envs))
        print(fourier_plot.environment_detector(rate, signal2n, *envs))

    # Plot environments.
    frequencies = np.fft.rfftfreq(256, 1 / rate)
    for i, env in enumerate(envs):
        plt.scatter(frequencies, env, s=1, label=places[i])
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


if __name__ == '__main__':
    main()
