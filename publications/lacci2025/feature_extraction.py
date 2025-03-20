import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import rfft
import librosa


def rms_value(signal: np.ndarray):
    """
    root mean square
    """
    rms = np.sqrt(np.mean(signal**2))
    return rms


def sra_value(signal: np.ndarray):
    """
    square root of the amplitude
    """
    sra = (np.mean(np.sqrt(np.absolute(signal)))) ** 2
    return sra


def kv_value(signal: np.ndarray):
    """
    kurtosis value
    """
    kv = kurtosis(signal)
    return kv


def sv_value(signal: np.ndarray):
    """
    skewness value
    """
    sv = skew(signal)
    return sv


def ppv_value(signal: np.ndarray):
    """
    peak to peak value
    """
    return np.max(signal) - np.min(signal)


def cf_value(signal: np.ndarray):
    """
    crest factor
    """
    cf = np.max(np.absolute(signal)) / rms_value(signal)
    return cf


def if_value(signal: np.ndarray):
    """
    impulse value
    """
    _if = np.max(np.absolute(signal)) / np.mean(np.absolute(signal))
    return _if


def mf_value(signal: np.ndarray):
    """
    margin factor
    """
    mf = np.max(np.absolute(signal)) / sra_value(signal)
    return mf


def sf_value(signal: np.ndarray):
    """
    shape factor
    """
    sf = rms_value(signal) / np.mean(np.absolute(signal))
    return sf


def kf_value(signal: np.ndarray):
    """
    kurtosis factor
    """
    return kv_value(signal) / (rms_value(signal) ** 4)


def fc_value(signal: np.ndarray):
    """
    Frequency center
    """
    fft_normalized = 2 * np.abs(rfft(signal)) / signal.size
    return np.mean(fft_normalized)


def rmsf_value(signal: np.ndarray):
    """
    Root mean square frequency
    """
    fft_normalized = 2 * np.abs(rfft(signal)) / signal.size
    return np.sqrt(np.mean(fft_normalized**2))


def rvf_value(signal: np.ndarray):
    """
    Root variance frequency
    """
    fft_normalized = 2 * np.abs(rfft(signal)) / signal.size
    return np.sqrt(np.mean((fft_normalized - fc_value(signal)) ** 2))


def _get_features_from_signal(signal: np.ndarray[float]) -> np.ndarray[float]:
    attributes = []
    attributes.append(rms_value(signal))
    attributes.append(sra_value(signal))
    attributes.append(kv_value(signal))
    attributes.append(sv_value(signal))
    attributes.append(ppv_value(signal))
    attributes.append(cf_value(signal))
    attributes.append(if_value(signal))
    attributes.append(mf_value(signal))
    attributes.append(sf_value(signal))
    attributes.append(kf_value(signal))
    attributes.append(fc_value(signal))
    attributes.append(rmsf_value(signal))
    attributes.append(rvf_value(signal))
    return np.array(attributes, dtype=float)


class StatAndSpectralFeatures:
    def __init__(self, windows=1):
        self._windows = windows

    def fit_transform(self, files: list[str]) -> np.ndarray[float]:
        files_amount = len(files)
        num_features_per_window = 13
        features_per_file = num_features_per_window * self._windows
        result = np.zeros((files_amount, features_per_file), dtype=float)

        for i, file in enumerate(files):
            signal, _ = librosa.load(file, sr=None)
            splited_signal = np.array_split(signal, self._windows)
            for j, s in enumerate(splited_signal):
                features = _get_features_from_signal(s)
                begin_index = j * num_features_per_window
                end_index = begin_index + num_features_per_window
                result[i, begin_index:end_index] = features

        return result


if __name__ == "__main__":
    files = [
        "C:\\Users\\LeonardoBortoni\\Documents\\Ich\\Mestrado\\mtsa-tests\\0_dB_fan\\fan\\id_02\\normal\\00000012.wav",
        "C:\\Users\\LeonardoBortoni\\Documents\\Ich\\Mestrado\\mtsa-tests\\0_dB_fan\\fan\\id_02\\normal\\00000022.wav",
    ]

    featurer = StatAndSpectralFeatures(1)
    a = featurer.fit_transform(files)
    b = 2
