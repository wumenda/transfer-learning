import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_sne(data, labels):
    # 使用t-SNE对数据进行降维
    tsne = TSNE(n_components=2, random_state=0)
    data_tsne = tsne.fit_transform(data)
    # 创建一个新的图形
    fig = plt.figure(figsize=(8, 8))
    fig.patch.set_alpha(0.7)
    # 绘制 x=0 和 y=0 的虚线
    plt.axhline(0, color="gray", linestyle="dashed")
    plt.axvline(0, color="gray", linestyle="dashed")
    # 获取唯一的标签值
    unique_labels = np.unique(labels)
    # 为每个标签分配不同的颜色
    # colors = plt.cm.tab20c(np.linspace(0, 1, len(unique_labels)))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    # 绘制不同颜色的数据点
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            data_tsne[mask, 0],
            data_tsne[mask, 1],
            s=5,
            label=f"Label {label}",
            color=colors[i],
        )
    plt.title("t-SNE Visualization")
    plt.legend(
        loc="upper right",
        fontsize="small",
        ncol=2,
        frameon=True,
    )


def plot(data, labels):
    tsne = TSNE(n_components=2, random_state=0)
    data_tsne = tsne.fit_transform(data)
    # 绘制 t-SNE 图
    plt.figure(figsize=(12, 8))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap="viridis")
    plt.colorbar()
    plt.title("t-SNE Visualization of Prediction")
    plt.legend()
    plt.show()


# 示例用法
if __name__ == "__main__":
    # 在这里替换成你的数据和标签
    your_data = np.random.rand(100, 10)  # 示例随机数据，你需要用你的数据替换它
    your_labels = np.random.randint(0, 3, 100)  # 示例随机标签，你需要用你的标签替换它
    plot_sne(your_data, your_labels)
