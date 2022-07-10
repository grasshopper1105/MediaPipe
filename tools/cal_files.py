import os
import numpy as np
import pandas as pd
import torch
from numpy import float32


def get_frame_idx(_csv_data, numElems=4):
    return np.round(np.linspace(0, _csv_data.shape[0] - 1, numElems)).astype(int)


def get_one_dataset(_csv_path, _frame):
    _csv_data = pd.read_csv(_csv_path, header=None, sep=',').iloc[1:, 1:]  # 不读取标题行

    _label = torch.from_numpy(np.array(_csv_data.iloc[0, 0]).astype(int))

    _csv_data_no_label = _csv_data.iloc[:, 1:]
    _csv_data_cut = _csv_data_no_label.iloc[get_frame_idx(_csv_data_no_label, _frame), :]  # 取出需要的数据
    _feature = torch.from_numpy(np.array(_csv_data_cut).astype(float32))

    # print(_label, _feature.shape)
    # _mean = torch.mean(_feature, dim=0)
    # _std = torch.std(_feature, dim=0)
    # _feature = (_feature - _mean) / _std

    # _label = torch.from_numpy(np.array(_csv_data_cut.iloc[0, 0]).astype(int))
    return _feature, _label


def make_idx(_idx, _sum_list):
    _file_dir_idx = np.argmax(np.array(_sum_list) > _idx)
    _file_idx = _idx - _sum_list[_file_dir_idx - 1]

    return _file_dir_idx, _file_idx


def calculate_files_num(_path, _class):
    _sum = 0
    _file_sum_list = []
    for _ in range(0, _class):
        path = _path + str(_).rjust(3, '0')
        files = os.listdir(path)  # 读入文件夹
        _num = len(files)  # 统计文件夹中的文件个数
        _sum += _num
        _file_sum_list.append(_sum)
    return _sum, _file_sum_list


DATA_PATH = "/root/hy-tmp/data/data/"
FRAME = 20  # 截取帧数
NUM_CLASS = 70  # 手语词类
FILE_LEN, FILE_SUM_LIST = calculate_files_num(DATA_PATH, NUM_CLASS)  # 总共的文件数、每个文件夹文件数
FIRST_FILE_NUM = FILE_SUM_LIST[0]  # 第一个文件夹的文件数
TRAIN_SPLIT = 0.75  # 训练集比例
EPOCH = 500  # 训练轮数
BATCH_SIZE = 4  # batch大小
HIDDEN_SIZE = 256  # 隐藏层
INPUT_SIZE = 92  # 特征数
LR = 5e-5  # 学习率
DROP_OUT = 0.1  # 随机抛弃
NUM_LAYER = 2  # RNN层数

for index in range(FILE_LEN):
    if index < FIRST_FILE_NUM:
        _file_dir_idx = 0
        _file_idx = index
    else:
        _file_dir_idx, _file_idx = make_idx(index, FILE_SUM_LIST)
    path = DATA_PATH + str(_file_dir_idx).rjust(3, '0') + "/" + str(_file_idx).rjust(3, '0') + ".csv"
    # path = self.path + str(_file_dir_idx).rjust(3, '0') + "\\" + str(_file_idx).rjust(3, '0') + ".csv"
    data, label = get_one_dataset(path, FRAME)
    print(data.shape)
