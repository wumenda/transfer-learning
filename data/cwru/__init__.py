import os
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, TensorDataset, random_split
from tqdm import tqdm

from util.transforms import compute_fft

__cwru_class__ = ["N", "F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]


class Crwu(Dataset):
    def __init__(
        self,
        root: str,
        num_classes: int,
        sample_num: int,
        sample_length: int,
        rate: list,
        feature="de",
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
    def get_file_paths(root):
        # 获取文件夹中所有的 .mat 文件路径
        file_paths = []
        for file_name in os.listdir(root):
            if file_name.endswith(".mat"):
                file_path = os.path.join(root, file_name)
                file_paths.append(file_path)
        return file_paths

    def load_one_mat(self, file_path):
        # 加载 .mat 文件
        data = loadmat(file_path)
        for key, value in data.items():
            if "DE" in key:
                de = torch.squeeze(torch.tensor(value, dtype=torch.float32))
            elif "FE" in key:
                fe = torch.squeeze(torch.tensor(value, dtype=torch.float32))
        if self.feature == "all":
            data = torch.vstack((de, fe))
        elif self.feature == "de":
            data = torch.vstack((de,))
        elif self.feature == "fe":
            data = torch.vstack((fe,))
        sample = self.sampling_slice(data, self.sample_length, self.sample_num)
        label = torch.full(
            (sample.shape[0],),
            int(os.path.splitext(os.path.basename(file_path))[0]),
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
        mat_paths = self.get_file_paths(root)
        x = []
        y = []
        for i, path in tqdm(enumerate(mat_paths)):
            sample, label = self.load_one_mat(path)
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
    source_root = os.path.join(args.root, f"{args.task.split('-')[0]}HP")
    cwru = Crwu(
        source_root,
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
    cwru = Crwu(
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
    source_dataset = Crwu(
        source_root,
        args.num_classes,
        args.sample_num,
        args.sample_length,
        args.rate,
        feature=args.feature,
        fft=args.fft,
    )
    target_dataset = Crwu(
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
