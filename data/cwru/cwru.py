import os
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, TensorDataset, random_split
from tqdm import tqdm

from util.transforms import compute_fft

__cwru_class__ = ["N", "F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]


class Cwru:
    def __init__(
        self,
        root: str,
        num_classes: int,
        sample_num: int,
        sample_length: int,
        rate: float,
        feature="de",
        fft=False,
    ):
        self.num_classes = num_classes
        self.sample_num = sample_num
        self.sample_length = sample_length
        self.rate = 0.7
        self.feature = feature
        self.fft = fft
        self.dataset_train, self.dataset_val = self.get_data(root)

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
        sample_train, sample_val = self.sampling_slice(
            data, self.sample_length, self.sample_num
        )
        label_train = torch.full(
            (sample_train.shape[0],),
            int(os.path.splitext(os.path.basename(file_path))[0]),
            dtype=torch.long,
        )
        label_val = torch.full(
            (sample_val.shape[0],),
            int(os.path.splitext(os.path.basename(file_path))[0]),
            dtype=torch.long,
        )
        return sample_train, sample_val, label_train, label_val

    def sampling_slice(self, data: torch.Tensor, sample_length, sample_num):
        data_length = data.shape[1]
        interval = (data_length - sample_length) // sample_num
        slice_datas_train = []
        slice_datas_val = []
        start, end = 0, sample_length
        for i in range(0, sample_num):
            slice_data = data[:, start:end]
            if self.fft is True:
                slice_data = torch.fft.fft(slice_data, dim=1)
            slice_data_abs = torch.abs(slice_data)
            if i < sample_num * self.rate:
                slice_datas_train.append(slice_data_abs)
            else:
                slice_datas_val.append(slice_data_abs)
            start, end = start + interval, end + interval
        # 将切片后的数据转换为张量形式
        train_sample = torch.stack(slice_datas_train)
        val_sample = torch.stack(slice_datas_val)
        # 返回数据
        return train_sample, val_sample

    def get_data(self, root):
        mat_paths = self.get_file_paths(root)
        x_train = []
        y_train = []
        x_val = []
        y_val = []
        for i, path in tqdm(enumerate(mat_paths)):
            sample_train, sample_val, label_train, label_val = self.load_one_mat(path)
            x_train.append(sample_train)
            y_train.append(label_train)
            x_val.append(sample_val)
            y_val.append(label_val)
        samples_train = torch.cat(x_train, dim=0)
        labels_train = torch.cat(y_train, dim=0)
        samples_val = torch.cat(x_val, dim=0)
        labels_val = torch.cat(y_val, dim=0)
        return TensorDataset(samples_train, labels_train), TensorDataset(
            samples_val, labels_val
        )

    # def divide_set(self):
    #     return random_split(self.dataset, self.rate)

    @property
    def one_hot_matrix(self):
        return torch.eye(self.num_classes)


def build_dataset(args):
    source_root = os.path.join(args.root, f"{args.task.split('-')[0]}HP")
    cwru = Cwru(
        source_root,
        args.num_classes,
        args.sample_num,
        args.sample_length,
        args.rate,
        feature=args.feature,
        fft=args.fft,
    )
    train_set, test_set = (
        cwru.dataset_train,
        cwru.dataset_val,
    )
    return train_set, test_set


def build_test(args, root):
    cwru = Cwru(
        root,
        args.num_classes,
        args.sample_num,
        args.sample_length,
        args.rate,
        feature=args.feature,
        fft=args.fft,
    )
    return cwru.dataset_val


def build_transfer_task(args):
    source_root = os.path.join(args.root, f"{args.task.split('-')[0]}HP")
    target_root = os.path.join(args.root, f"{args.task.split('-')[1]}HP")
    source_dataset = Cwru(
        source_root,
        args.num_classes,
        args.sample_num,
        args.sample_length,
        args.rate,
        feature=args.feature,
        fft=args.fft,
    )
    source_train_set, source_val_set = (
        source_dataset.dataset_train,
        source_dataset.dataset_val,
    )
    target_dataset = Cwru(
        target_root,
        args.num_classes,
        args.sample_num,
        args.sample_length,
        args.rate,
        feature=args.feature,
        fft=args.fft,
    )
    taregt_train_set, target_val_set = (
        target_dataset.dataset_train,
        target_dataset.dataset_val,
    )
    return source_train_set, source_val_set, taregt_train_set, target_val_set
