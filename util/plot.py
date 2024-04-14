import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
from matplotlib.font_manager import FontProperties
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


font = FontProperties(fname="C:/Windows/Fonts/simhei.ttf", size=12)

plt.style.use("science")


# Configure Matplotlib to use Chinese fonts
plt.rcParams["font.family"] = [
    "SimSun",
    "Microsoft YaHei",
    "Arial",
]  # Use SimSun, Microsoft YaHei, or Arial for Chinese


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
    plt.savefig(save_path, bbox_inches="tight")


def plot(acc_list, save_path):
    size = 4
    x = np.arange(size)
    a = acc_list[0]
    b = acc_list[1]
    c = acc_list[2]
    d = acc_list[3]
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
    plt.savefig(save_path, bbox_inches="tight")


def plot_curve(
    data_list,
    save_path,
    title="accuracy ",
    xlabel="X",
    ylabel="Y",
):
    """
    绘制曲线图函数

    参数:
    - data_list (list): 包含要绘制的数据的列表
    - title (str): 图表的标题 (默认为 "Curve Plot")
    - xlabel (str): x 轴标签 (默认为 "X")
    - ylabel (str): y 轴标签 (默认为 "Y")
    """
    # 生成 x 轴的数据，假设从 0 开始，步长为 1
    window_size = 50
    x = list(range(len(data_list)))
    smoothed_y = np.convolve(
        data_list, np.ones(window_size) / window_size, mode="valid"
    )
    # 创建图表和子图
    plt.figure(figsize=(10, 6))

    # 计算平滑的趋势走向曲线
    smoothed_x = x[window_size // 2 : -window_size // 2 + 1]
    smoothed_x.insert(0, x[0])
    smoothed_y = np.insert(smoothed_y, 0, data_list[0])

    # 绘制原始曲线（半透明的灰色）
    plt.plot(x, data_list, color="#B3C8CF", alpha=0.5, label="original")

    # 绘制平滑的趋势走向曲线（加粗的克莱因蓝色）
    plt.plot(
        smoothed_x,
        smoothed_y,
        color="#135D66",
        linewidth=2,
        label="smooth",
    )

    # 设置 x 轴的范围，将原点置于左边框
    plt.xlim(0, len(data_list) - 1)

    plt.legend()

    # 添加标题和标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 显示图表
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()()


def plot_muti_curve(
    data_list,
    save_path,
    title="accuracy ",
    xlabel="X",
    ylabel="Y",
):
    """
    绘制多条曲线的图表函数

    参数:
    - data_list (list): 包含要绘制的数据序列的二维列表
    - save_path (str): 图像保存路径
    - title (str): 图表的标题 (默认为 "accuracy")
    - xlabel (str): x 轴标签 (默认为 "X")
    - ylabel (str): y 轴标签 (默认为 "Y")
    """
    # 生成 x 轴的数据，假设从 0 开始，步长为 1
    x = list(range(max(len(data) for data in data_list)))

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 循环遍历每个数据序列并绘制曲线
    for idx, data in enumerate(data_list):
        # 生成平滑的趋势走向曲线
        window_size = 50
        smoothed_y = np.convolve(data, np.ones(window_size) / window_size, mode="valid")
        smoothed_x = x[window_size // 2 : -window_size // 2 + 1]
        smoothed_x.insert(0, x[0])
        smoothed_y = np.insert(smoothed_y, 0, data[0])

        # 绘制原始曲线
        plt.plot(x, data, color=f"C{idx}", alpha=0.5, label=f"original {idx+1}")

        # 绘制平滑的趋势走向曲线
        plt.plot(
            smoothed_x,
            smoothed_y,
            color=f"C{idx}",
            linewidth=2,
            linestyle="--",
            label=f"smooth {idx+1}",
        )

    # 添加标题和标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    # 保存图像并显示
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()()


# def plot_confusion_matrix(
#     cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
# ):
#     """
#     - cm : 计算出的混淆矩阵的值
#     - classes : 混淆矩阵中每一行每一列对应的列
#     - normalize : True:显示百分比, False:显示个数
#     """
#     if normalize:
#         cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
#         print("显示百分比：")
#         np.set_printoptions(formatter={"float": "{: 0.2f}".format})
#         print(cm)
#     else:
#         print("显示具体数字：")
#         print(cm)
#     plt.imshow(cm, interpolation="nearest", cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#     # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
#     plt.ylim(len(classes) - 0.5, -0.5)
#     fmt = ".2f" if normalize else "d"
#     thresh = cm.max() / 2.0
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(
#             j,
#             i,
#             format(cm[i, j], fmt),
#             horizontalalignment="center",
#             color="white" if cm[i, j] > thresh else "black",
#         )
#     plt.tight_layout()
#     plt.ylabel("True label")
#     plt.xlabel("Predicted label")
#     # plt.show()()


def draw_matrix(predict, labels, classes, save_path):
    cm = confusion_matrix(predict, labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, cmap="Blues", fmt="d", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()()


if __name__ == "__main__":
    day_time = datetime.now().strftime("%Y-%m-%d_%H_%M")
    acc = [
        [0.97, 0.59, 0.68, 0.51],
        [0.56, 0.93, 0.79, 0.73],
        [0.60, 0.81, 0.93, 0.65],
        [0.56, 0.76, 0.73, 0.91],
    ]
    a = [1, 2, 3, 4, 5, 0.97, 0.59, 0.68, 0.51]
    # plot(acc, f"acc-{day_time}.png")
    plot_curve(a)
