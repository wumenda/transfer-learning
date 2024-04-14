import numpy as np
import pywt
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def plot_signal_spectrum(signal, sample_rate):
    # 计算信号的长度
    signal_len = len(signal)

    # 对信号进行FFT变换
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(signal_len, 1 / sample_rate)

    # 只绘制正频率部分（去除负频率）
    positive_freqs = fft_freqs[: signal_len // 2]
    magnitude_spectrum = np.abs(fft_vals[: signal_len // 2])

    # 绘制频谱图
    plt.figure(figsize=(8, 6))
    plt.plot(positive_freqs, magnitude_spectrum)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude Spectrum")
    plt.title("Signal Spectrum")
    plt.grid(True)
    # plt.show()()


def wavelet_transform(signal, wavelet="db4", level=5):
    # 获取离散小波对象
    wavelet = pywt.Wavelet(wavelet)

    # 进行小波变换
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # 提取近似系数和细节系数
    cA = coeffs[0]
    cD = coeffs[1:]

    # 计算近似系数和细节系数的长度
    len_cA = len(cA)
    len_cD = [len(c) for c in cD]

    # 构建时频图像
    time_freq_img = np.zeros((level + 1, len(signal)))

    # 将近似系数填充到时频图像的底部
    time_freq_img[0, :len_cA] = cA

    # 将细节系数填充到时频图像的上部
    for i, cd in enumerate(cD):
        time_freq_img[i + 1, : len_cD[i]] = cd

    return time_freq_img


# 示例使用
# 假设有一个长度为N的振动信号
N = 1000
signal = np.random.randn(N)

# 进行小波变换，得到时频图像
wavelet_img = wavelet_transform(signal)

# 绘制时频图像
plt.imshow(wavelet_img, aspect="auto", cmap="jet")
plt.colorbar()
plt.title("Wavelet Transform: Time-Frequency Image")
plt.xlabel("Time")
plt.ylabel("Scale")
# plt.show()()

plot_signal_spectrum(signal, 100)
