import numpy as np
import pywt


def continuous_wavelet_transform(signal, wavelet="morl", scales=np.arange(1, 11)):
    # 进行连续小波变换
    coefs, freqs = pywt.cwt(signal, scales, wavelet)

    return coefs, freqs


# 示例使用
# 假设有一个长度为N的信号
N = 1000
time = np.arange(N)
signal = (
    np.sin(2 * np.pi * 2 * time / N)
    + np.sin(2 * np.pi * 10 * time / N)
    + np.random.randn(N)
)

# 设定尺度范围
scales = np.arange(1, 11)

# 进行连续小波变换，得到系数和频率
coefs, freqs = continuous_wavelet_transform(signal, scales=scales)

# 可视化连续小波系数
import matplotlib.pyplot as plt

plt.imshow(coefs, cmap="jet", aspect="auto")
plt.colorbar()
plt.title("Continuous Wavelet Coefficients")
plt.xlabel("Time")
plt.ylabel("Scale")
# plt.show()()

# 可视化频率谱
plt.plot(freqs, np.abs(coefs.mean(axis=1)), "b")
plt.title("Average Power Spectrum")
plt.xlabel("Frequency")
plt.ylabel("Power")
# plt.show()()
