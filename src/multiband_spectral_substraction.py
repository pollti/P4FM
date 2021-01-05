from typing import Tuple

import numpy as np
from numpy import fft


def SSMultibandKamath02(signal: np.ndarray, fs: int, IS: float = .25) -> np.ndarray:  # TODO: use noise detection replacing IS
    """
    Multi-band Spectral subtraction. Subtraction with adjusting subtraction factor. The adjustment is according to local a postriori SNR and the frequency band.
    :param signal: noisy initial signal
    :param fs: sampling frequency
    :param IS: initial silence length. Will be replaced later (TODO)
    :return: denoised signal. CAUTION: Maybe slightly shorter than input.
    """
    W = int(.025 * fs)  # 25 ms sequences # TODO: replace by noise detection interval
    nfft = W
    SP = .4  # Shift percentage is 40% (10ms)
    wnd = np.hamming(W)

    NIS = int((IS * fs - W) / (SP * W) + 1)  # number of initial silence segments # TODO: replace everywhere NIS
    Gamma = 2  # Magnitude Power (1 for magnitude spectral subtraction 2 for power spectrum subtraction)

    # y = statsmodels.tsa.ar_model.AutoReg(unknown, unknown2).fit()
    y = segment(signal, W, SP, wnd)
    Y = np.fft.fft(y.T, nfft).T  # .Ts after debug
    YPhase = np.angle(Y[0:int((Y.shape[0] / 2)) + 1, :])  # Noisy Speech Phase
    Y = np.power(np.abs(Y[0:int((Y.shape[0] / 2)) + 1, :]), Gamma)
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
    Delta[0:int((-2000 + fs / 2) * FreqResol * 2 / fs)] = 2.5  # if the frequency is lower than FS/2 - 2KHz
    Delta[0:int(1000 * FreqResol * 2 / fs)] = 1  # if the frequency is lower than 1KHz
    X = np.zeros((FreqResol, numberOfFrames))
    for i in range(0, numberOfFrames):
        [NoiseFlag, SpeechFlag, NoiseCounter, Dist] = vad(np.power(Y[:, i], (1 / Gamma)), np.power(N, (1 / Gamma)), NoiseCounter)  # Magnitude Spectrum Distance VAD
        if SpeechFlag == 0:
            N = (NoiseLength * N + Y[:, i]) / (NoiseLength + 1)  # Update and smooth noise
            BN = Beta * N

        SNR = 10 * np.log(Y[:, i] / N)
        alpha = alphaSlope * SNR + alphaShift
        alpha = np.maximum(np.minimum(alpha, maxalpha), minalpha)

        D = Y[:, i] - np.multiply(np.multiply(Delta, alpha), N)  # Nonlinear (Non-uniform) Power Spectrum Subtraction

        X[:, i] = np.maximum(D, BN)  # if BY>D X=BY else X=D which sets very small values of subtraction result to an attenuated version of the input power spectrum.

    output = OverlapAdd2(np.power(X, (1 / Gamma)), YPhase, W, int(SP * W)).astype(np.int16)

    return output


def OverlapAdd2(xnew: np.ndarray, yphase: np.ndarray = None, window_len: int = None, shift_len: int = None) -> np.ndarray:
    """
    Reconstructs the signal.
    :param xnew: two dimensional array holding a fft of a signal segment per column
    :param yphase: phase angle of spectrum. Dimension is identical to xnew's. Defaulting to phase angle of xnew (for real values: zero)
    :param window_len: window length of time domain segments. Defaulting to twice fft window length
    :param shift_len: shift length of segmentation length (window_len if no overlap, lower for overlap). Defaulting to 50% overlap (window_len/2).
    :return: signal reconstructed from given spectrogram
    """
    if yphase is None:
        yphase = np.angle(xnew)
    if window_len is None:
        window_len = xnew.shape[0] * 2
    if shift_len is None:
        shift_len = window_len / 2

    (freq_res, frame_num) = xnew.shape
    Spec = xnew * np.exp(1j * yphase)

    if window_len % 2:  # window length is odd
        Spec = np.r_[Spec, np.flipud(np.conj(Spec[1:, :]))]
    else:
        Spec = np.r_[Spec, np.flipud(np.conj(Spec[1:-1, :]))]

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
    spectral_dist = np.clip(spectral_dist, 0, None)  # cut off negative values
    dist = np.mean(spectral_dist).astype(float)
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
    segment_number = int((signal_length - samples_per_window) / sp + 1)
    a = np.tile(np.arange(samples_per_window), (segment_number, 1))
    b = np.tile(np.arange(segment_number) * sp, (samples_per_window, 1)).T
    index = (a + b).T
    hw = np.tile(window, (segment_number, 1)).T

    return signal[index] * hw
