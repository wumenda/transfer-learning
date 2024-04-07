import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import pandas as pd

plt.style.use("science")


def plot_acc_bar(file_path, save_path):
    # 从 Excel 文件中读取数据
    df = pd.read_excel(file_path)

    # 设置 Source 列为索引
    df.set_index("Source", inplace=True)
    size = 4
    x = np.arange(size)
    a = df.iloc[0, 0:].to_list()
    b = df.iloc[1, 0:].to_list()
    c = df.iloc[2, 0:].to_list()
    d = df.iloc[3, 0:].to_list()
    tasks = [[f"T{i}{j}" for j in range(4)] for i in range(4)]

    total_width, n = 0.8, 4
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.figure(figsize=(8, 6))
    plt.bar(x, a, width=width, label="a")
    plt.bar(x + width, b, width=width, label="b")
    plt.bar(x + 2 * width, c, width=width, label="c")
    plt.bar(x + 3 * width, d, width=width, label="d")
    plt.xticks(
        np.concatenate([x, x + width, x + 2 * width, x + 3 * width]),
        np.concatenate(tasks),
    )
    plt.legend()
    plt.savefig(save_path)
