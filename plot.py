import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.family"] = ["SimSun", "Microsoft YaHei", "Arial"]

if __name__ == "__main__":
    data = [
        [0.77, 0.76, 0.67, 0.89],
        [0.94, 0.89, 0.9, 0.92],
        [0.86, 0.78, 0.89, 0.95],
        [0.98, 0.95, 0.98, 0.98],
    ]
    size = 4
    x = np.arange(size)
    a, b, c, d = data

    tasks = [[f"T{i}" for j in range(4)] for i in range(4)]

    total_width, n = 0.8, 4
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.figure(figsize=(12, 6))
    plt.bar(x, a, width=width, label="WDCNN")
    plt.bar(x + width, b, width=width, label="DANN-W")
    plt.bar(x + 2 * width, c, width=width, label="DANN-T")
    plt.bar(x + 3 * width, d, width=width, label="Proposed Methed")
    plt.xticks(
        np.concatenate([x, x + width, x + 2 * width, x + 3 * width]),
        np.concatenate(tasks),
    )

    # 调整图例位置与大小
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(0.1, 1.1),
        prop={"size": 10},
        ncol=4,  # 横向排布
    )

    plt.savefig("figurebar", bbox_inches="tight")
