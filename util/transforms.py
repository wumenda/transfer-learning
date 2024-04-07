import numpy as np
from scipy.fft import fft


def compute_fft(signal, sample_rate):
    # 计算信号的长度
    signal_length = len(signal)

    # 计算采样间隔
    sample_interval = 1 / sample_rate

    # 生成时间轴
    time_axis = np.arange(0, signal_length) * sample_interval

    # 计算信号的 FFT
    signal_fft = fft(signal)

    # 计算频率轴
    frequency_axis = np.linspace(0, sample_rate, signal_length)

    return signal_fft
