import numpy as np
import matplotlib.pyplot as plt
import pywt

# 生成示例信号
fs = 1000  # 采样率（Hz）
t = np.linspace(0, 1, fs, endpoint=False)  # 时间轴
f1, f2 = 50, 100  # 信号中的两个频率成分
signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

# 连续小波变换的参数
wavelet = "morl"  # 选择小波基函数
scales = np.arange(1, 128)  # 尺度参数的范围

# 进行连续小波变换
coefficients, frequencies = pywt.cwt(signal, scales, wavelet)

# 绘制频谱图
plt.figure(figsize=(10, 5))
plt.imshow(
    abs(coefficients),
    extent=[0, 1, frequencies[-1], frequencies[0]],
    aspect="auto",
    cmap="jet",
)
plt.colorbar(label="幅度")
plt.xlabel("时间（秒）")
plt.ylabel("频率（Hz）")
plt.title("连续小波变换频谱图")
plt.show()
