from typing import Tuple

import numpy as np
from numpy import fft


# Created with the help of the following Matlab implementation:
#  Esfandiar Zavarehei (2021). Multi-band Spectral Subtraction (https://www.mathworks.com/matlabcentral/fileexchange/7674-multi-band-spectral-subtraction), MATLAB Central File Exchange. Retrieved February 19, 2021.

# Related paper for technical details: Kamath, Sunil, and Philipos Loizou. "A multi-band spectral subtraction method for enhancing speech corrupted by colored noise." ICASSP. Vol. 4. 2002.

def multiband_substraction_denoise(signal: np.ndarray, fs: int, a: int = 0, b: int = 2560) -> np.ndarray:
    """
    Multi-band Spectral subtraction. Subtraction with adjusting subtraction factor. The adjustment is according to local a postriori SNR and the frequency band.
    :param signal: noisy initial signal
    :param fs: sampling frequency (samples per second)
    :param a: first sample index of largest noise only segment
    :param b: last sample index of largest noise only segment
    :return: denoised signal. CAUTION: Maybe slightly shorter than input.
    """
    segment_size = int(.025 * fs)  # 25 ms sequences
    nfft = segment_size
    shift = .4  # Segment shift percentage is 40% (10ms)
    wnd = np.hamming(segment_size)

    first_silence_segment = int(a / (shift * segment_size) + 1)
    last_silence_segment = int((b - segment_size) / (shift * segment_size) + 1)
    if last_silence_segment <= first_silence_segment:
        print("Denoising error (multipass): to few noise found. Assuming first .25 seconds to be noise.")
        first_silence_segment = 0
        last_silence_segment = 10
    # NIS = int((init_silence * fs - segment_size) / (shift * segment_size) + 1)  # number of initial silence segments; replaced by detected noise interval
    gamma = 2  # Magnitude Power (1 for magnitude spectral subtraction 2 for power spectrum subtraction)

    # y = statsmodels.tsa.ar_model.AutoReg(unknown, unknown2).fit()
    y = segment(signal, segment_size, shift, wnd)
    y_fft = np.fft.fft(y.T, nfft).T  # .Ts after debug
    y_phase = np.angle(y_fft[0:int((y_fft.shape[0] / 2)) + 1, :])  # Noisy Speech Phase
    y_fft = np.power(np.abs(y_fft[0:int((y_fft.shape[0] / 2)) + 1, :]), gamma)
    numberOfFrames = y_fft.shape[1]
    FreqResol = y_fft.shape[0]

    N = np.mean(y_fft[:, first_silence_segment:last_silence_segment], 1).T  # initial Noise Power Spectrum mean

    noise_counter = 0  # consecutive noises
    noise_length = 9  # This is a smoothing factor for the noise updating

    # See paper for detailed explainations.
    beta = .03
    minalpha = 1
    maxalpha = 5
    minSNR = -5
    maxSNR = 20
    alphaSlope = (minalpha - maxalpha) / (maxSNR - minSNR)
    alphaShift = maxalpha - alphaSlope * minSNR
    BN = beta * N

    # Delta is a frequency dependent coefficient that can be adjusted here.
    delta = 1.5 * np.ones(BN.size)
    delta[0:int((-2000 + fs / 2) * FreqResol * 2 / fs)] = 2.5  # if the frequency is lower than FS/2 - 2KHz
    delta[0:int(1000 * FreqResol * 2 / fs)] = 1  # if the frequency is lower than 1KHz
    X = np.zeros((FreqResol, numberOfFrames))
    for i in range(0, numberOfFrames):
        [noise_flag, speech_flag, noise_counter, dist] = vad(np.power(y_fft[:, i], (1 / gamma)), np.power(N, (1 / gamma)), noise_counter)  # Magnitude Spectrum Distance VAD
        if speech_flag == 0:
            N = (noise_length * N + y_fft[:, i]) / (noise_length + 1)  # Update and smooth noise
            BN = beta * N

        SNR = 10 * np.log(y_fft[:, i] / N)
        alpha = alphaSlope * SNR + alphaShift
        alpha = np.maximum(np.minimum(alpha, maxalpha), minalpha)

        D = y_fft[:, i] - np.multiply(np.multiply(delta, alpha), N)  # Nonlinear (Non-uniform) Power Spectrum Subtraction

        X[:, i] = np.maximum(D, BN)  # if BY>D X=BY else X=D which sets very small values of subtraction result to an attenuated version of the input power spectrum.

    output = signal_reconstruction(np.power(X, (1 / gamma)), y_phase, segment_size, int(shift * segment_size)).astype(np.int16)

    return output


def signal_reconstruction(xnew: np.ndarray, yphase: np.ndarray = None, window_len: int = None, shift_len: int = None) -> np.ndarray:
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
    spec = xnew * np.exp(1j * yphase)

    if window_len % 2:  # window length is odd
        spec = np.r_[spec, np.flipud(np.conj(spec[1:, :]))]
    else:  # even window length
        spec = np.r_[spec, np.flipud(np.conj(spec[1:-1, :]))]

    sig = np.zeros((frame_num - 1) * shift_len + window_len)

    for i in np.arange(frame_num):
        start = i * shift_len
        spec_i = spec[:, i]
        sig[start:start + window_len] += np.real(np.fft.ifft(spec_i, window_len))

    # returning a signal from frequencies
    return sig


def vad(signal: np.ndarray, noise: np.ndarray, noise_counter: int = 0, noise_margin: int = 3, hangover: int = 8) -> Tuple[int, int, int, float]:
    """
    Spectral Distance Voice Activity Detector.
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

    # simple maxtrix operations for reordering
    signal_length = signal.size
    sp = int(samples_per_window * shift_percentage)
    segment_number = int((signal_length - samples_per_window) / sp + 1)
    a = np.tile(np.arange(samples_per_window), (segment_number, 1))
    b = np.tile(np.arange(segment_number) * sp, (samples_per_window, 1)).T
    index = (a + b).T
    # hamming window application
    hw = np.tile(window, (segment_number, 1)).T

    return signal[index] * hw
