import os
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, TensorDataset, random_split
from tqdm import tqdm

from util.transforms import compute_fft


class Pu(Dataset):
    def __init__(
        self,
        root: str,
        num_classes: int,
        sample_num: int,
        sample_length: int,
        rate: list,
        feature="vibration",
        fft=False,
    ):
        self.num_classes = num_classes
        self.sample_num = sample_num
        self.sample_length = sample_length
        self.rate = rate
        self.feature = feature
        self.fft = fft
        self.dataset = self.get_data(root)

    @staticmethod
    def get_all_folders(directory):
        # 存储所有文件夹路径的列表
        folder_paths = []

        # 遍历指定目录下的所有文件夹
        for root, dirs, files in os.walk(directory):
            # 遍历文件夹列表
            for folder in dirs:
                # 构建文件夹的完整路径
                folder_path = os.path.join(root, folder)
                # 将文件夹路径添加到列表中
                folder_paths.append(folder_path)

        return folder_paths

    @staticmethod
    def get_file_paths(root):
        # 获取文件夹中所有的 .mat 文件路径
        file_paths = []
        for file_name in os.listdir(root):
            if file_name.endswith(".mat"):
                file_path = os.path.join(root, file_name)
                file_paths.append(file_path)
        return file_paths

    def load_one_mat(self, file_path, label):
        # 加载 .mat 文件
        data = loadmat(file_path, squeeze_me=True)
        filename = os.path.split(file_path)[1].split(".")[0]
        vibration = data[filename]["Y"].item()[6]["Data"]
        if self.feature == "vibration":
            vibration = torch.squeeze(torch.tensor(vibration, dtype=torch.float32))
        elif self.feature == "force":
            pass
        if self.feature == "vibration":
            data = torch.vstack((vibration,))
        sample = self.sampling_slice(data, self.sample_length, self.sample_num)
        label = torch.full(
            (sample.shape[0],),
            label,
            dtype=torch.long,
        )
        return sample, label

    def sampling_slice(self, data: torch.Tensor, sample_length, sample_num):
        data_length = data.shape[1]
        interval = (data_length - sample_length) // sample_num
        slice_datas = []
        start, end = 0, sample_length
        for i in range(0, sample_num):
            slice_data = data[:, start:end]
            if self.fft is True:
                slice_data = torch.fft.fft(slice_data, dim=0)
            slice_datas.append(slice_data.real)
            start, end = start + interval, end + interval
        # 将切片后的数据转换为张量形式
        sample = torch.stack(slice_datas)
        # 返回数据
        return sample

    def get_data(self, root):
        folders = self.get_all_folders(root)
        x = []
        y = []
        for i, folder in enumerate(folders):
            mat_paths = self.get_file_paths(folder)
            for j, path in tqdm(enumerate(mat_paths)):
                sample, label = self.load_one_mat(path, i)
                x.append(sample)
                y.append(label)
            samples = torch.cat(x, dim=0)
            labels = torch.cat(y, dim=0)
        return TensorDataset(samples, labels)

    def divide_set(self):
        return random_split(self.dataset, self.rate)

    @property
    def one_hot_matrix(self):
        return torch.eye(self.num_classes)


def build_dataset(args):
    cwru = Pu(
        args.root,
        args.num_classes,
        args.sample_num,
        args.sample_length,
        args.rate,
        feature=args.feature,
        fft=args.fft,
    )
    train_set, test_set = cwru.divide_set()
    return train_set, test_set


def build_test(args, root):
    cwru = Pu(
        root,
        args.num_classes,
        args.sample_num,
        args.sample_length,
        args.rate,
        feature=args.feature,
        fft=args.fft,
    )
    return cwru.dataset


def build_transfer_task(args):
    source_root = os.path.join(args.root, f"{args.task.split('-')[0]}HP")
    target_root = os.path.join(args.root, f"{args.task.split('-')[1]}HP")
    source_dataset = Pu(
        source_root,
        args.num_classes,
        args.sample_num,
        args.sample_length,
        args.rate,
        feature=args.feature,
        fft=args.fft,
    )
    target_dataset = Pu(
        target_root,
        args.num_classes,
        args.sample_num,
        args.sample_length,
        args.rate,
        feature=args.feature,
        fft=args.fft,
    )
    source_train_set, source_val_set = source_dataset.divide_set()
    taregt_train_set, target_val_set = target_dataset.divide_set()
    return source_train_set, source_val_set, taregt_train_set, target_val_set
