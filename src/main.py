import tempfile
from copy import deepcopy
from typing import Tuple

import scipy as scp
import scipy.signal
import scipy.io.wavfile
import numpy as np
import matplotlib_tuda
import noisereduce as nr
import progressbar
import webrtcvad
from sacred.observers import FileStorageObserver

import fourier_plot
from matplotlib import pyplot as plt, ticker
from matplotlib.contour import QuadContourSet

# from src.multiband_spectral_substraction import segment
from src.plot_util import SubplotsAndSave
from src.multiband_spectral_substraction import multiband_substraction_denoise
from src.aggregation_method import AggregationMethod

from sacred import Experiment

ex = Experiment()
ex.observers.append(FileStorageObserver("results"))

matplotlib_tuda.load()
np.set_printoptions(linewidth=2000)


# noinspection PyUnusedLocal
@ex.config
def ex_config():
    de_signaled = False  # set true to use pre designaled recordings AND recordings_to_be_assigned; Not recommended for most cases.
    # (not supported yet) environments_calculated = False  # set true to use precalculated environments

    # Audio settings
    path = "Audio/"  # The relative path to the recordings depends on working directory
    ending = ".wav"  # audio fileending, currently only supports .wav files
    ## Give noise_only files and environments here instead of the parameters above, if de_signaled = False - not recommended
    envs = None
    recordings_noise_only = None
    recordings_to_be_assigned_noise_only = None
    activate_multipass = False  # whether to use multipass approach for comparisions
    multipass_audio_files = activate_multipass  # whether to use multipass for audio file generation
    audio_save = True  # whether to generate the audio files (denoised, noise only) as well
    snooze = False  # no audio files or plots generated - may be much faster

    # Graph properties
    show_graphs_denoising = [True, True, True]  # what to show graphs: [original, denoised, noise only]
    filename_graph_denoising = "file_denoising_steps"  # filename for denoising graph
    denoising_graph_multipass = activate_multipass  # whether to use multipass for denoising plot
    filename_graph_comparing = "compare_denoised_files"  # filename for envoronment comparision graph
    comparing_graph_multipass = activate_multipass  # whether to use multipass for comparision plot

    # Comparision metric parameters
    squared: bool = True  # square errors (otherwise: linear)
    y_log: bool = False  # logarithmic y scaling in comparision
    x_log: bool = False  # logarithmic x scaling in comparision (has no effect for frequency_aggregation = MEDIAN)
    dga: bool = False  # Use Double Gaussian Approach
    frequency_aggregation_method: AggregationMethod = AggregationMethod.MEAN  # Aggregation method over frequencies in window
    window_aggregation_method: AggregationMethod = AggregationMethod.MEDIAN  # Aggregation method over different windows

@ex.named_config
def ex_config_example():
    recordings = {"Raum1": ["Fabi01", "Fabi02", "Fabi03", "Fabi04", "Fabi05"],
                  "Raum2": ["Tim01", "Tim02", "Tim03", "Tim04", "Tim05"],
                  "Platz": ["Platz01", "Platz02", "Platz03", "Platz04", "Platz05"],
                  "Strasse": ["Street01", "Street02", "Street03", "Street04", "Street05"],
                  "Treppe": ["Treppe01", "Treppe02", "Treppe03", "Treppe04", "Treppe05"],
                  "Wald": ["Wald01", "Wald02", "Wald03", "Wald04", "Wald05"]}  # filenames assigned to location
    recordings_to_be_assigned = deepcopy(
        recordings)  # deepcopy notwendig um nicht nur die Referenz zu kopieren, for this default case we want to assign the already assigned data to see how acurate it works
    recordings_to_be_assigned["unknown"] = ["Fabi01"]  # put one file for test purpose

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
    minnoise = 512  # minimal noise interval must be specified here to avoid crashes
    largest_size = 0
    start = 0
    end = minnoise
    for a, b in intervals:
        if b - a > largest_size:
            start = a
            end = b
            largest_size = b - a
    if end - start <= minnoise:
        print(
            "Warning: no long pure noise interval found. Assuming first milliseconds to be pure noise. Results may be highly imprecise.")
    return start, end


def denoise_audio(signal: np.ndarray, rate: int, including_multipass: bool = True) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, int, int]:
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
    reduced_noise = nr.reduce_noise(audio_clip=signal.astype(np.float16), noise_clip=noisy_part.astype(np.float16),
                                    verbose=False).astype(np.int16)
    noise_signal = signal - reduced_noise
    # Call multipass if needed
    if including_multipass:
        voice_leakage = multiband_substraction_denoise(noise_signal, rate, a, b)
        noise_signal = noise_signal[:voice_leakage.size] - voice_leakage
    return reduced_noise, noise_signal, intervals, a, b


@ex.capture
def read_audio(path_and_name: str, ending: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Reads Wave-Files.
    :param path: path to file from working directory folder, without the .wav ending
    :param ending: file format ending like ".wav"
    :returns: shape:(T,) all the sampled times as floats
    :returns: shape:(S,) the samples of the left stereoline as int16 values
    :returns: samplerate as float
    """
    rate, data = scipy.io.wavfile.read(f'{path_and_name}{ending}')
    time_step = 1 / rate
    time_vec = np.arange(0, data.shape[0]) * time_step
    if len(data.shape) == 2:
        data = data[:, 0]
    return time_vec, data, int(rate)


@ex.capture
def generate_exemplary_audio(endtime: int = 70, time_step: float = 0.01) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    :TODO: take params for signal functions for debugging purposes (currently hardcoded)
    Generates exemplary audio signal
    :returns: shape:(T,) all the sampled time    time_step = .01s as floats
    :returns: shape:(S,) the samples of the left stereoline as int16 values
    :returns: samplerate as float
    """
    time_vec = np.arange(endtime, step=time_step)
    # Generate linear chirp with f_0 = 1 and k = 0.2.
    signal = np.sin(0.5 * np.pi * time_vec * (1 + 0.1 * time_vec))
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
    # fig.show()


def plot_spectrogram(signal: np.ndarray, rate: float, caption: str, ax, show_xlabel: bool, show_ylabel: bool,
                     show_title: bool) -> QuadContourSet:
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

    c = ax.contourf(times, freqs, spectrogram, 10.0 ** np.arange(-6, 6), locator=ticker.LogLocator(),
                    cmap=plt.get_cmap('jet'))
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
def plot_denoise(filenames: np.ndarray, filename_graph_denoising: str, show_graphs_denoising,
                 denoising_graph_multipass: bool, path: str):
    """
    Generates required files (denoised, noise) and plots spectrograms.
    :param filenames: files to process as list. Files are rows.
    :param filename_graph_denoising: name of the saved plot file
    :param show_graphs_denoising: list of plots and files to generate as booleans. [orig, denoised, noise]
    :param denoising_graph_multipass: Apply multipass approach for better denoising.
    :param path: path to audio files.
    """
    print('Generating plots for specified files. This may take a while.')
    plottypes = show_graphs_denoising.count(True)
    if plottypes < 1:
        print("Warning: No plots selected for denoising information. Quitting plotting.")
        return
    # ncols + 1 for scala space. This generates an empty plot, but is the best solution for now.
    # CAUTION: if removed null pointers may occur later in axs[show_graph[:1].count(True)]
    with tempfile.TemporaryDirectory() as dirpath:
        with SubplotsAndSave(dirpath, filename_graph_denoising, nrows=len(filenames), ncols=plottypes + 1, sharey='all',
                             file_types=['png']) as (fig, axs):
            bar = progressbar.ProgressBar(
                widgets=['Creating plot: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()],
                maxval=plottypes * len(filenames)).start()
            for i, filename in enumerate(filenames):
                time_vec, signal, rate = read_audio(path + filename)
                #### START debugging shortener - ignore if not debugging. Used to reduce signal length for faster testing.
                # fact = 1
                # end = rate
                # signal = signal[::fact]
                # time_vec = time_vec[::fact]
                # rate = int(rate / fact)
                #### END debugging shortener

                # if only one file, axs is one-dimensional
                if axs.ndim > 1:
                    axs_i = [axs[i, 0], axs[i, + show_graphs_denoising[:1].count(True)],
                             axs[i, + show_graphs_denoising[:2].count(True)]]
                else:
                    axs_i = [axs[0], axs[show_graphs_denoising[:1].count(True)],
                             axs[show_graphs_denoising[:2].count(True)]]

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
                            ax.plot((time_vec[0], time_vec[interval_start]), (line_y, line_y), color='tuda:green',
                                    zorder=10)
                        # Assume no noise between intervals.
                        if k > 0:
                            ax.plot((time_vec[intervals[k - 1][1]], time_vec[interval_start]), (line_y, line_y),
                                    color='tuda:green', zorder=100)
                        # Assume no noise to end of time.
                        if k == intervals.shape[0] - 1:
                            ax.plot((time_vec[interval_end], time_vec[-1]), (line_y, line_y), color='tuda:green',
                                    zorder=10)
                        ax.plot((time_vec[interval_start], time_vec[interval_end]), (line_y, line_y), color='tuda:red',
                                zorder=100)
                    ax.plot((time_vec[b], time_vec[a]), (line_y, line_y), color='black', zorder=100)

                    bar.update(i * plottypes + 1)

                # Generate denoised signal if needed.
                if show_graphs_denoising[1] or show_graphs_denoising[2]:
                    denoised_signal, noise_signal, _, _, _ = denoise_audio(signal, rate,
                                                                           including_multipass=denoising_graph_multipass)

                # Denoised signal.
                if show_graphs_denoising[1]:
                    # scipy.io.wavfile.write(f'media/{filename}_denoised_generated.wav', rate, denoised_signal)
                    # plot_signal(time_vec, denoised_signal, filename + '_den')
                    c = plot_spectrogram(denoised_signal, rate, 'denoised', axs_i[1], i == plottypes - 1,
                                         not show_graphs_denoising[0], i == 0)
                    bar.update(i * plottypes + show_graphs_denoising[:1].count(True) + 1)

                # Noise only.
                if show_graphs_denoising[2]:
                    # scipy.io.wavfile.write(f'media/{filename}_noiseonly_generated.wav', rate, noise_signal)
                    # plot_signal(time_vec, noise_signal, filename + '_noise')
                    c = plot_spectrogram(noise_signal, rate, 'noise', axs_i[2], i == plottypes - 1,
                                         not (show_graphs_denoising[0] | show_graphs_denoising[1]), i == 0)
                    bar.update(i * plottypes + show_graphs_denoising[:2].count(True) + 1)

            bar.finish()
            print('Saving files to disk. Please stand by.')
            cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])  # left, bottom, width, height in percent
            fig.colorbar(c, cax=cbar_ax)
            # fig.show()

        ex.add_artifact(f'{dirpath}/{filename_graph_denoising}.png', name=filename_graph_denoising + '.png')


@ex.capture
def plot_place_comparision(recordings: dict, filename_graph_comparing: str, comparing_graph_multipass: bool, path: str):
    """
    :param recordings: A mapping of locations to filenames to be plotted.
    :param filename_graph_comparing: name of the saved plot file
    :param comparing_graph_multipass: Apply multipass approach for better denoising.
    :param path: path to audio files.
    Plots a comparision of denoised spectrums for environments with same number of files for each.
    """

    print('Generating plots for places. This may take a while.')

    c = None
    # Recording places
    places_count = len(recordings)
    # Recording ids
    items_count = max([len(n) for n in recordings.values()])
    # ncols + 1 for scala space. This generates an empty plot, but is the best solution for now.
    # CAUTION: if removed null pointers may occur later in axs[show_graph[:1].count(True)]
    with tempfile.TemporaryDirectory() as dirpath:
        with SubplotsAndSave(dirpath, filename_graph_comparing, nrows=items_count, ncols=places_count + 1, sharey='all',
                             file_types=['png']) as (fig, axs):
            bar = progressbar.ProgressBar(
                widgets=['Creating plot: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()],
                maxval=items_count * places_count).start()
            for i, place in enumerate(recordings.keys()):
                for j, rec in enumerate(recordings[place]):
                    time_vec, signal, rate = read_audio(f'{path}{rec}')

                    denoised_signal, noise_signal, intervals, a, b = denoise_audio(signal, rate,
                                                                                   comparing_graph_multipass)
                    # scipy.io.wavfile.write(f'{path}{filename}_denoised_generated.wav', rate, denoised_signal)
                    # scipy.io.wavfile.write(f'{path}{filename}_noiseonly_generated.wav', rate, noise_signal)

                    # if only one recording for every environment, axs is one-dimensional
                    ax = axs[j, i] if items_count > 1 else axs[i]
                    c = plot_spectrogram(noise_signal, rate, place + ', Noise', ax, j == len(recordings[place]) - 1,
                                         i == 0, j == 0)

                    # Noise and speach lines at plot top.
                    line_y = 9980
                    # Colorize intervals of pe noise / speech.
                    for k, (interval_start, interval_end) in enumerate(intervals):
                        # Assume no noise from beginning of time.
                        if k == 0:
                            ax.plot((time_vec[0], time_vec[interval_start]), (line_y, line_y), color='tuda:green',
                                    zorder=10)
                        # Assume no noise between intervals.
                        if k > 0:
                            ax.plot((time_vec[intervals[k - 1][1]], time_vec[interval_start]), (line_y, line_y),
                                    color='tuda:green', zorder=100)
                        # Assume no noise to end of time.
                        if k == intervals.shape[0] - 1:
                            ax.plot((time_vec[interval_end], time_vec[-1]), (line_y, line_y), color='tuda:green',
                                    zorder=10)
                        ax.plot((time_vec[interval_start], time_vec[interval_end]), (line_y, line_y), color='tuda:red',
                                zorder=100)

                    ax.plot((time_vec[b], time_vec[a]), (line_y, line_y), color='black', zorder=100)
                    bar.update(i * items_count + j + 1)
            bar.finish()
            print('Saving files to disk. Please stand by.')
            cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])  # left, bottom, width, height in percent
            fig.colorbar(c, cax=cbar_ax)
            # fig.show()

        ex.add_artifact(f'{dirpath}/{filename_graph_comparing}.png', name=filename_graph_comparing + '.png')


@ex.capture
def save_audio_files(recordings: dict, path: str, multipass_audio_files: bool, text: str = ''):
    """
    :param recordings: A mapping of locations to filenames to be processed.
    :param path: Path to audio files.
    :param multipass_audio_files: Activate multipass.
    :param text: Additional text for filename beginning. Should end with "_".
    Generates audio files for original, denoising, designalized. The latter two can use multipass.
    """

    print('Generating audio files. This may take a while.')

    for place, recs in recordings.items():
        for rec in recs:
            with tempfile.TemporaryDirectory() as dirpath:
                time_vec, signal, rate = read_audio(f'{path}{rec}')

                # Denoise audio.
                denoised_signal, noise_signal, intervals, a, b = denoise_audio(signal, rate, multipass_audio_files)

                # Geate Audio files.
                filenames = [f'{dirpath}/{text}{place}_{rec}_{itm}.wav' for itm in
                             ['original', 'denoised' + ('mp' if multipass_audio_files else ''), 'noiseonly']]
                scipy.io.wavfile.write(filenames[0], rate, signal)
                scipy.io.wavfile.write(filenames[1], rate, denoised_signal)
                scipy.io.wavfile.write(filenames[2], rate, noise_signal)

                # Save audio files.
                for f in filenames:
                    ex.add_resource(f)


@ex.automain
def main(snooze: bool, recordings: dict, recordings_to_be_assigned: dict, path: str, ending: str, envs: np.ndarray, de_signaled: bool, frequency_aggregation_method: AggregationMethod,
         window_aggregation_method: AggregationMethod, recordings_to_be_assigned_noise_only: dict, activate_multipass: bool, audio_save: bool, plot_environment: bool = True,
         plot_denoising_spectrums: bool = True, squared: bool = True, y_log: bool = False, x_log: bool = False, dga: bool = False):  # TODO documentation; parse parameters
    """
    :param snooze: Switch off plots and audio.
    :param recordings: A mapping of locations to filenames to be processed.
    :param recordings_to_be_assigned: A mapping of estimated locations to filenames. The algorithm will search for the location of the respective files.
    :param path: The relative path from the working directory to the audio files
    :param ending: The ending of the audiofiles.
    :param envs: Already computed and saved environments.
    :param de_signaled: If there are already designaled and saved audiofiles.
    :param frequency_aggregation_method: The Aggregation Method used for frequencies.
    :param window_aggregation_method: The Aggregation Method used for the windows.
    :param recordings_to_be_assigned_noise_only: A mapping of location to filenames of files, which are noise-only.
    :param activate_multipass: If the multipass method will be used.
    :param audio_save: If the audio files are to be saved.
    :param plot_environment: If the environments are to be plotted.
    :param plot_denoising_spectrums: If the denoising spectrums are to be plotted.
    :param squared: If the square error is to be computed. (Instead of the linear error)
    :param y_log: If the the y values are to be exchanged by their logarithm values before the difference calculation.
    :param x_log: If the the y values are to be exchanged by their logarithm values before the difference calculation.
    :param dga: Double Gaussion Approach DGA weights frequencies in common voice spectrum lower. Use with caution as manipulations in voice spectrum might not be detected and results may be less good anyway.
    """
    # time_vec, signal, rate = read_audio('tmp')
    # segment(signal[:44100], 256)
    envs = {}  # Mapping from place to calculated environment
    rate = 44100  # default value avoids crashes when plotting
    
    # Step 1: Read audio files for environments & denoise/devoice
    if not de_signaled:
        recordings_noise_only = {}
        recordings_to_be_assigned_noise_only = {}
        for place, recs in recordings.items():
            signals = []
            designaled_signals = []
            for rec in recs:
                _, signal, rate = read_audio(path + rec)
                signals.append(signal)
                _, designaled_signal, _, _, _ = denoise_audio(signal, rate, activate_multipass)
                designaled_signals.append(designaled_signal)

            # Step 2: Generate environments as ndarray
            envs[place] = fourier_plot.environment_generator(designaled_signals, window_aggregation_method)
            recordings_noise_only[place] = designaled_signals

            # Step 3: Optional: plot denoising spectrums
            if plot_denoising_spectrums & (not snooze):
                plot_denoise(recs, filename_graph_denoising=f'file_denoising_steps_environment_{place}')

        # Step 4: Optional plot environment spectrums
        if plot_environment & (not snooze):
            plot_place_comparision(recordings, filename_graph_comparing=f'compare_denoised_files_environments')

        # Step 5: Optional: save audio files for environments
        if audio_save & (not snooze):
            save_audio_files(recordings, text='environment_')

        # Step 6: Denoise/devoice audio to be assigned
        for place, recs in recordings_to_be_assigned.items():
            signals = []
            designaled_signals = []
            for rec in recs:
                _, signal, rate = read_audio(path + rec)
                signals.append(signal)
                _, designaled_signal, _, _, _ = denoise_audio(signal, rate, activate_multipass)
                designaled_signals.append(designaled_signal)

            recordings_to_be_assigned_noise_only[place] = designaled_signals

            # Step 7: Optional: plot denoising spectrums
            if plot_denoising_spectrums & (not snooze):
                plot_denoise(recs, filename_graph_denoising=f'file_denoising_steps_assigning_{place}')

            # Step 8: Optional: save audio files for recordings to be assigned
            if audio_save & (not snooze):
                save_audio_files(recordings, text='recording_')

    # Step 9: Detect environments and saves into file
    with tempfile.TemporaryDirectory() as dirpath:
        filename = "environment_errors.csv"
        with open(f'{dirpath}/{filename}', "w") as file:
            file.write("recorded_in," + ",".join(["error_to_" + env for env in envs.keys()]) + "\n")
            for place, denoised_signals in recordings_to_be_assigned_noise_only.items():
                for denoised_signal in denoised_signals:
                    tmp = ",".join(
                        [str(x) for x in
                         fourier_plot.environment_detector(rate, denoised_signal, frequency_aggregation_method, squared, y_log, x_log, dga, *envs.values())])
                    temp = place + "," + tmp
                    file.write(temp + "\n")
        ex.add_artifact(f'{dirpath}/{filename}', name=filename)

    if not snooze:
        # Step X: Plot environments.
        frequencies = np.fft.rfftfreq(256, 1 / rate)
        for place, env in envs.items():
            plt.scatter(frequencies, env, s=1, label=place)
        # for row in all_data.T:
        #    plt.scatter(frequencies, row, s=1)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim((rate / 256, rate / 2 + 1))
        plt.ylim((10 ** 1, 10 ** 5))
        plt.legend(loc="upper right")
        plt.xlabel("frequency")
        plt.ylabel("energy")
        plt.title("Environment data")
        with tempfile.TemporaryDirectory() as dirpath:
            filename = "environments_fourier"
            plt.savefig(f'{dirpath}/{filename}.png')
            ex.add_artifact(f'{dirpath}/{filename}.png', name=filename + '.png')
        # plt.show()
