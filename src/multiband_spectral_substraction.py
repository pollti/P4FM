from typing import Tuple

import numpy as np
from numpy import fft


# import statsmodels


def SSMultibandKamath02(signal: np.ndaray, fs, IS: float = .25) -> np.ndarray:  # TODO: use noise detection replacing IS
    W = int(.025 * fs)  # 25 ms sequences # TODO: replace by noise detection interval
    nfft = W
    SP = .4  # Shift percentage is 40% (10ms)
    wnd = np.hamming(W)

    NIS = int((IS * fs - W) / (SP * W) + 1)  # number of initial silence segments # TODO: replace everywhere NIS
    Gamma = 2  # Magnitude Power (1 for magnitude spectral subtraction 2 for power spectrum subtraction)

    # TODO: unimplemented part: https://viewer.mathworks.com/?viewer=plain_code&url=https%3A%2F%2Fde.mathworks.com%2Fmatlabcentral%2Fmlc-downloads%2Fdownloads%2Fsubmissions%2F7674%2Fversions%2F1%2Fcontents%2FSSMultibandKamath02.m

    unknown = []
    unknown2 = 4
    # y = statsmodels.tsa.ar_model.AutoReg(unknown, unknown2).fit()  # TODO fix
    y = segment(signal, W, SP, wnd)
    Y = np.fft.fft(y, nfft)
    YPhase = np.angle(Y[0:(Y.size / 2).floor + 1, :])  # Noisy Speech Phase # TODO: off-by-1?
    Y = np.power(np.abs(Y[0:(Y.size / 2).floor + 1, :]), Gamma)  # TODO: off-by-1?
    numberOfFrames = Y.shape[1]
    FreqResol = Y.shape[0]

    N = np.mean(Y[:, 0: NIS], 1).T  # initial Noise Power Spectrum mean

    NoiseCounter = 0
    NoiseLength = 9  # This is a smoothing factor for the noise updating

    Beta = .03
    minalpha = 1
    maxalpha = 5
    minSNR = -5
    maxSNR = 20
    alphaSlope = (minalpha - maxalpha) / (maxSNR - minSNR)
    alphaShift = maxalpha - alphaSlope * minSNR
    BN = Beta * N

    # Delta is a frequency dependent coefficient
    Delta = 1.5 * np.ones(BN.size)
    Delta[0:((-2000 + fs / 2) * FreqResol * 2 / fs).floor] = 2.5;  # if the frequency is lower than FS/2 - 2KHz # TODO: ob1?
    Delta[0:(1000 * FreqResol * 2 / fs).floor] = 1;  # if the frequency is lower than 1KHz # TODO: ob1?
    for i in range(0, numberOfFrames):
        [NoiseFlag, SpeechFlag, NoiseCounter, Dist] = vad(np.power(Y[:, i], (1 / Gamma)), np.power(N, (1 / Gamma)), NoiseCounter)  # Magnitude Spectrum Distance VAD
        if SpeechFlag == 0:
            N = (NoiseLength * N + Y[:, i]) / (NoiseLength + 1)  # Update and smooth noise
            BN = Beta * N

        SNR = 10 * np - np.log(Y[:, i]) / N
        alpha = alphaSlope * SNR + alphaShift
        alpha = np.max(np.min(alpha, maxalpha), minalpha)

        D = Y[:, i] - np.multiply(np.multiply(Delta, alpha), N)  # Nonlinear (Non-uniform) Power Spectrum Subtraction

        X[:, i] = np.max(D, BN);  # if BY>D X=BY else X=D which sets very small values of subtraction result to an attenuated version of the input power spectrum.

    output = OverlapAdd2(np.pow(X, (1 / Gamma)), YPhase, W, SP * W);

    return output


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
