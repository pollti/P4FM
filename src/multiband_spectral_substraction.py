from typing import Tuple

import numpy as np


def SSMultibandKamath02(signal: np.ndaray, FS, IS: float = .25) -> np.ndarray:  # TODO: use noise detection replacing IS
    W = int(.025 * FS)  # 25 ms sequences # TODO: replace by noise detection interval
    nfft = W
    SP = .4  # Shift percentage is 40% (10ms)
    wnd = np.hamming(W)

    NIS = int((IS * FS - W) / (SP * W) + 1)  # number of initial silence segments # TODO: replace
    Gamma = 2 # Magnitude Power (1 for magnitude spectral subtraction 2 for power spectrum subtraction)

    # TODO: unimplemented part: https://viewer.mathworks.com/?viewer=plain_code&url=https%3A%2F%2Fde.mathworks.com%2Fmatlabcentral%2Fmlc-downloads%2Fdownloads%2Fsubmissions%2F7674%2Fversions%2F1%2Fcontents%2FSSMultibandKamath02.m

    return signal


def OverlapAdd2(xnew: np.ndarray, yphase: np.ndarray = None, window_len: int = None, shift_len: int = None) -> np.ndarray:
    """

    :param xnew: two dimensional array holding a fft of a signal segment per column
    :param yphase: phase angle of spectrum. Dimension is identical to xnew's. Defaulting to phase angle of xnew (for real values: zero)
    :param window_len: window length of time domain segments. Defaulting to twice fft window length
    :param shift_len: shift length of segmentation length (window_len if no overlap, lower for overlap). Defaulting to 50% overlap (window_len/2).
    :return: signal reconstructed from given spectrogram
    """
    if yphase is None:
        yphase = xnew  # TODO
    if window_len is None:
        window_len = 0  # TODO
    if shift_len is None:
        shift_len = window_len / 2

    (freq_res, frame_num) = xnew.shape

    Spec = xnew * np.exp(1j * yphase)

    if (window_len % 2) == 0:  # window length is odd. TODO: might need freq_res for window_length
        Spec = np.append(Spec, np.flipud(np.conj(Spec[1:, :])))
    else:
        Spec = np.append(Spec, np.flipud(np.conj(Spec[1:-1, :])))

    sig = np.zeros((frame_num - 1) * shift_len + window_len)

    for i in np.arange(frame_num):
        start = i * shift_len
        spec = Spec[:, i]
        sig[start:start + window_len] += np.real(np.fft.ifft(spec, window_len))

    return sig


def vad(signal: np.ndarray, noise: np.ndarray, noise_counter: int = 0, noise_margin: int = 3, hangover: int = 8) -> Tuple[int, int, int, float]:
    """
    Spectral Distance Vioce Activity Detector.
    :param signal: magnitude spectrum of current frame to be {noise,speech} labeled
    :param noise: magnitude spectrum of noise as estimated
    :param noise_counter: number of noise frames directly preceding. Default: 0 (no noise directly before)
    :param noise_margin: spectral distance value. Default: 3. Lower values will more likely classify as speech.
    :param hangover: number of noise segments to reset speechflag after
    :return noise_flag:  1: noise, 0: speech
    :return speech_flag:  reset after hangover noise segments
    :return noise_counter: number of previous noise segments. Increased by 1 or reset to 0.
    :return dist:  spectral distance of signal here and expected noise
    """
    # freq_resol = signal.size
    spectral_dist = 20 * (np.log10(signal) - np.log10(noise))
    spectral_ist = np.clip(spectral_dist, 0)  # cut off negative values
    dist = np.mean(spectral_dist)[0]
    if dist < noise_margin:
        noise_flag = 1
        noise_counter += 1
    else:
        noise_flag = 0
        noise_counter = 0
    # Detect noise only periods and attenuate the signal.
    if noise_counter > hangover:
        speech_flag = 0
    else:
        speech_flag = 1
    return noise_flag, speech_flag, noise_counter, dist


def segment(signal: np.ndarray, samples_per_window: int = 256, shift_percentage: float = 0.4, window: np.ndarray = None) -> np.ndarray:  # Last parameter default value
    """
    Chop signal to overlapping segments with given window and overlap.
    :param signal: the input signal, apply on reasonable sized intervals only to avoid RAM crashes
    :param samples_per_window: size of single windows. Default: 256
    :param shift_percentage: percentage of shift per new window in percents. Default: 40%
    :param window: Window function to multiply with each interval (as array). Default: Hamming window application. Size must be equal to samples per window.
    :return: chopped signal as two-dimensional numpy array
    """
    if window is None:
        window = np.hamming(samples_per_window)  # Hamming window als Fensterfunktion: https://de.wikipedia.org/wiki/Fensterfunktion
    signal_length = signal.size
    sp = int(samples_per_window * shift_percentage)
    segment_number = int((signal_length - samples_per_window) / sp + 1)  # TODO: check +1
    a = np.tile(np.arange(samples_per_window), (segment_number, 1))
    b = np.tile(np.arange(segment_number) * sp, (samples_per_window, 1)).T
    index = (a + b).T
    hw = np.tile(window, (segment_number, 1)).T

    return signal[index] * hw
