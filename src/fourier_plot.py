from deprecated import deprecated
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from src.multiband_spectral_substraction import segment
from src.aggregation_method import AggregationMethod


def environment_generator(signals, window_avg, size: int = 256) -> np.ndarray:
    """
    Generates a noise expectation for an environment based on many recordings.
    :param signals: The signals of the recordings from the given environment. Should be noise only already by here and not be too short/few.
    :param window_avg: Choice whether to use median or mean.
    :param size: The window size for a single rfft. Choose higher values for more detailed frequency comparision. Must be identical to environment detection size later.
    :return: frequency-energy expectation for this environment.
    """
    all_data = None
    for signal in signals:
        signal = segment(signal, size, 0.4, sp.signal.gaussian(size, size * 0.4))
        data = np.abs(np.fft.rfft(signal, axis=0))
        all_data = data if all_data is None else np.hstack((all_data, data))
    if window_avg == AggregationMethod.MEDIAN:
        data = np.median(all_data, axis=1)
    else:
        data = np.mean(all_data, axis=1)
    return data


def environment_detector(rate: int, signal: np.ndarray, frequency_avg: AggregationMethod, squared: bool, y_log: bool, x_log: bool, dga: bool, *envs: np.ndarray, size: int = 256) -> np.ndarray:
    """
    Detects environment plausibility for a given recorded signal and several environments. Result dimension is highly dependent on squared error paramters.
    :param rate: Sample rate of all signals.
    :param signal: The signal of the recording under testing. Should be noise only already by here and not be too short.
    :param frequency_avg: Choice whether to use median or mean.
    :param squared: Use Mean Square Error MSE instead of absolute differences. Weights higher differences more.
    :param y_log: Applies mean error to logarithmated energy values, so punctual divergencies have less influence.
    :param x_log: Weights low x frequencies higher resulting in an (about) equal weighting of all octaves in spectrum.
    :param dga: Double Gaussion Approach DGA weights frequencies in common voice spectrum lower. Use with caution as manipulations in voice spectrum might not be detected and results may be less good anyway.
    :param envs: Expected frequency-energy levels per environment. Same window size as [size] and sample rate as [rate] must have been used to generate these.
    :param size: The window size for a single rfft. Choose higher values for more detailed frequency comparision.
    :return: Difference values per environment in the order of environments provided as parameters.
    """
    signal = segment(signal, size, 0.4, sp.signal.gaussian(size, size * 0.4))
    data = np.abs(np.fft.rfft(signal, axis=0))
    # windows = data.shape[1]
    env_values = None
    for env in envs:
        errors = None
        for row in data.T:
            # any metric can be used here, scalar or ndarray. In case of ndarray: Caution with median later! Performance may be poor with arrays.
            tmp = error_mean(row, env, frequency_avg, squared, y_log, x_log, dga, rate, size)
            errors = tmp if errors is None else np.hstack((errors, tmp))
        env_value = np.array([np.median(errors)])
        env_values = env_value if env_values is None else np.hstack((env_values, env_value))
    return env_values


def error_mean(a: np.ndarray, b: np.ndarray, frequency_avg: AggregationMethod, squared: bool, y_log: bool, x_log: bool, dga: bool, rate: int, size: int):
    """
    A flexible function for computing different kinds of mean errors of two arrays. Intentionally used for frequency-energy-levels.
    :param a: First array (intentionally an environment).
    :param b: Second array (intentionally an unassociated environment).
    :param frequency_avg: Choice whether to use median or mean.
    :param squared: Use Mean Square Error MSE instead of absolute differences. Weights higher differences more.
    :param y_log: Applies mean error to logarithmated energy values, so punctual divergencies have less influence.
    :param x_log: Weights low x frequencies higher resulting in an (about) equal weighting of all octaves in spectrum.
    :param dga: Double Gaussion Approach DGA weights frequencies in common voice spectrum lower. Use with caution as manipulations in voice spectrum might not be detected and results may be less good anyway.
    :param rate: Sample rate of all signals.
    :param size: The window size for a single rfft. FFT is not applied here, but results in different length arrays. Could be replaced by analyzing a or b.
    :return:
    """
    if x_log and dga:
        print("Parameters incompatible: x_log and dga. Unexpected effects may occur.")
    if y_log:
        a = np.log(a)
        b = np.log(b)
    err = error(a, b, squared)
    if y_log:
        err = np.exp(err)  # necessary? Yes.
    weights = np.ones(a.size)
    if x_log:
        weights = weights / sp.fft.rfftfreq(size, 1 / rate)
        weights[0] = 1
    if dga:
        freq = sp.fft.rfftfreq(size, 1 / rate)
        weights = sp.stats.norm.pdf(freq, 45, 20) + sp.stats.norm.pdf(freq, 1700, 4000)
        err = err * weights
    # x_log has no effect for median, so np.median sufficies then
    return np.sum(err) / np.sum(weights) if frequency_avg == AggregationMethod.MEAN else np.median(err)


def error(a: np.ndarray, b: np.ndarray, squared: bool):
    """
    Computes the absolute difference of two arrays of same dimension.
    :param a: First array (intentionally an environment).
    :param b: Second array (intentionally an unassociated environment).
    :param squared: square difference values.
    :return: array of differences with same dimensions as input arrays. Contains positive values only.
    """
    if a.shape != b.shape:
        print("Squared error: different dimensions – Crashing.")
    return np.square(a - b) if squared else np.abs(a - b)


@deprecated("Only used for debugging purposes. See plot in main for environment plot.")
def plot_fourier(rate: int, *signals: np.ndarray, size: int = 256):
    """
    Plots a frequency-energy diagram for the provided signals. Energy levels are meaned, one color is used per given signal. May also be used to see environments. Plots are not saved to disk.
    :param rate: Sample rate of all signals.
    :param signals: The signals to plot as numpy arrays. Must have at least [size] samples each, but should be way longer.
    :param size: The window size for a single rfft.
    """
    all_data = None
    frequencies = np.fft.rfftfreq(size, 1 / rate)
    for signal in signals:
        signal = segment(signal, size, 0.4, sp.signal.gaussian(size, size * 0.4))
        data = np.fft.rfft(signal, axis=0)

        data = np.abs(data)
        # data = np.sqrt(data ** 2)
        all_data = data if all_data is None else np.hstack((all_data, data))

        data = np.mean(data, 1)

        plt.scatter(frequencies, data, s=1)
    data = np.mean(all_data, axis=1)
    plt.scatter(frequencies, data, s=5)
    # for row in all_data.T:
    #    plt.scatter(frequencies, row, s=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((rate / size, rate / 2 + 1))
    plt.ylim((10 ** 1, 10 ** 5))
    # plt.ylim(ymin=0)
    # plt.show()
