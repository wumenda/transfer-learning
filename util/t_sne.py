import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scienceplots
from matplotlib.font_manager import FontProperties

font = FontProperties(fname="C:/Windows/Fonts/simhei.ttf", size=12)

plt.style.use("science")


def plot_sne(data, labels, savepath=""):
    # Using t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, random_state=0)
    data_tsne = tsne.fit_transform(data)

    # Create a new figure
    fig = plt.figure(figsize=(6, 6))
    fig.patch.set_alpha(0.7)
    ax = fig.add_subplot(111)

    # Plot dashed lines at x=0 and y=0
    plt.axhline(0, color="gray", linestyle="dashed", linewidth=0.5)
    plt.axvline(0, color="gray", linestyle="dashed", linewidth=0.5)

    # Get unique label values
    unique_labels = np.unique(labels)

    # Assign different colors and shapes for each label
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    markers = ["o", "s", "^", "x", "D", "P", "*", "h", "+", "v"]  # Define marker shapes

    # Plot data points with different colors and shapes
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            data_tsne[mask, 0],
            data_tsne[mask, 1],
            s=20,  # Adjust size of points
            label=f"Label {label}",
            color=colors[i],
            marker=markers[i % len(markers)],  # Cycle through markers
            edgecolor="black",  # Add black edges to markers
            linewidth=0.5,  # Set edge linewidth
            alpha=0.7,  # Set transparency
        )

    # Set title and legend
    plt.title("t-SNE Visualization", fontsize=14)
    plt.legend(loc="upper right", fontsize=8, frameon=False, markerscale=1.5)

    # Set grid style
    plt.grid(True, linestyle="--", alpha=0.5)

    # Set axis labels and ticks
    ax.set_xlabel("t-SNE Component 1", fontsize=10)
    ax.set_ylabel("t-SNE Component 2", fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Remove top and right spines
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    # plt.show()()


def plot(data, labels):
    tsne = TSNE(n_components=2, random_state=0)
    data_tsne = tsne.fit_transform(data)
    # 绘制 t-SNE 图
    plt.figure(figsize=(12, 8))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap="viridis")
    plt.colorbar()
    plt.title("t-SNE Visualization of Prediction")
    plt.legend()
    # plt.show()()


# 示例用法
if __name__ == "__main__":
    # 在这里替换成你的数据和标签
    your_data = np.random.rand(100, 10)  # 示例随机数据，你需要用你的数据替换它
    your_labels = np.random.randint(0, 3, 100)  # 示例随机标签，你需要用你的标签替换它
    plot_sne(your_data, your_labels)
