import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
x = np.linspace(0, 10, 1000)
y = np.sin(x) + np.random.normal(0, 0.4, size=x.shape)

# 计算滑动平均
window_size = 50
smoothed_y = np.convolve(y, np.ones(window_size) / window_size, mode="valid")

# 绘制原始曲线
plt.plot(x, y, label="Original Curve")

# 绘制平滑的趋势走向曲线
plt.plot(
    x[window_size // 2 : -window_size // 2 + 1],
    smoothed_y,
    color="red",
    label="Smoothed Curve",
)

plt.legend()
plt.show()
