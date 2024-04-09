import os

import pandas as pd
from datetime import datetime
import torch
import pandas as pd


def accuracy(outputs, labels):
    """
    Compute the accuracy
    outputs, labels: (tensor)
    return: (float) accuracy in [0, 100]
    """
    pre = torch.max(outputs.cpu(), 1)[1].numpy()
    y = labels.cpu().numpy()
    acc = (pre == y).sum() / len(y)
    return acc


class ExcelWriter:
    def __init__(self, file_path, header):
        """
        初始化 ExcelWriter 类。

        参数:
        - file_path: 要写入的 Excel 文件路径
        - header: Excel 文件的列名（header），以列表形式提供
        """
        self.file_path = file_path
        try:
            self.data_frame = pd.read_excel(file_path)
        except Exception as e:
            self.data_frame = pd.DataFrame(columns=header)
        self.header = list(self.data_frame.columns)

    def add_row(self, data_list: list, date=False):
        """
        将列表数据添加到 Excel 的新行中。

        参数:
        - data_list: 要添加到新行的列表数据，应该与 header 中的列名对应
        """
        if date is True:
            data_list.insert(0, get_date_now())
        if len(data_list) != len(self.header):
            raise ValueError("Length of data_list does not match the length of header")
        self.data_frame.loc[len(self.data_frame)] = data_list
        self.save()

    def save(self):
        """
        保存 Excel 文件。
        """
        self.data_frame.to_excel(self.file_path, index=False)
        print("数据已保存到 Excel 文件:", self.file_path)


def get_date_now():
    # 获取当前日期和时间
    now = datetime.now()
    # 格式化日期和时间，包括秒
    formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime


def add_column(value, args):
    table = pd.read_excel("predict_result.xlsx")
    table.loc[len(table)] = value
    # 将表格写回文件
    table.to_excel("predict_result.xlsx", index=False)


def write_list_to_file(data_list, file_path):
    """
    将列表写入文件，并以逗号分隔。

    参数:
    - data_list: 要写入文件的列表
    - file_path: 文件路径
    """
    # 将列表转换为以逗号分隔的字符串
    data_str = ",".join(data_list)

    # 将字符串写入文件
    with open(file_path, "w") as f:
        f.write(data_str)

    print("列表已写入到文件:", file_path)


if __name__ == "__main__":
    # 定义表格
    excel_writer = ExcelWriter("result.xlsx", ["日期", "PH0", "PH1", "PH2", "PH3"])
    data_list_1 = ["apple", "watermelon", "banana", "orange", "grape"]
    excel_writer.add_row(data_list_1)
