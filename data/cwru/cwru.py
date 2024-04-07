import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat


class CWRUBearingDataset(Dataset):
    def __init__(self, data_dir, window_size, stride, transform=None):
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".mat"):
                    file_path = os.path.join(root, file)
                    samples.append(file_path)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        sample = loadmat(sample_path)
        data = sample["data"]
        label = sample["label"]
        sliced_data = self._slice_data(data)
        if self.transform:
            sliced_data = self.transform(sliced_data)
        return sliced_data, label

    def _slice_data(self, data):
        num_slices = (data.shape[0] - self.window_size) // self.stride + 1
        sliced_data = []
        for i in range(num_slices):
            start = i * self.stride
            end = start + self.window_size
            sliced_data.append(data[start:end])
        return np.array(sliced_data)


# 示例转换函数，可根据需求进行修改
class ToTensor(object):
    def __call__(self, sample):
        data, label = sample
        return torch.from_numpy(data), torch.tensor(label)


# 使用示例
data_dir = "path_to_data_directory"
window_size = 1000  # 切片窗口大小
stride = 100  # 切片步长
transform = ToTensor()
dataset = CWRUBearingDataset(data_dir, window_size, stride, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
