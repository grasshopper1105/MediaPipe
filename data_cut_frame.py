import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from numpy import float32
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from RNNTest import RNN
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize': (20, 6)})


# def get_spaced_elements(array, numElems=4):
#     out = array[np.round(np.linspace(0, len(array) - 1, numElems)).astype(int)]
#     return out


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


def make_idx(_idx, _sum_list):
    _file_dir_idx = np.argmax(np.array(_sum_list) > _idx)
    _file_idx = _idx - _sum_list[_file_dir_idx - 1]

    return _file_dir_idx, _file_idx


# 超参数
# DATA_PATH = "D:/Software/PyCharm/AI/data/"
DATA_PATH = "/root/hy-tmp/data/data/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Use the GPU with', torch.cuda.get_device_name(0))

FRAME = 30  # 截取帧数
NUM_CLASS = 70  # 手语词类
FILE_LEN, FILE_SUM_LIST = calculate_files_num(DATA_PATH, NUM_CLASS)  # 总共的文件数、每个文件夹文件数
FIRST_FILE_NUM = FILE_SUM_LIST[0]  # 第一个文件夹的文件数
TRAIN_SPLIT = 0.75  # 训练集比例
EPOCH = 200  # 训练轮数
BATCH_SIZE = 64  # batch大小
HIDDEN_SIZE = 256  # 隐藏层
INPUT_SIZE = 92  # 特征数
LR = 5e-4  # 学习率
DROP_OUT = 0.1  # 随机抛弃
NUM_LAYER = 2  # RNN层数


class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, _csv_path, fps):  # 初始化一些需要传入的参数

        self.path = _csv_path
        self.data = None
        self.label = None
        self.frame = fps

    def __getitem__(self, index):  # 用于按照索引读取每个元素的具体内容

        if index < FIRST_FILE_NUM:
            _file_dir_idx = 0
            _file_idx = index
        else:
            _file_dir_idx, _file_idx = make_idx(index, FILE_SUM_LIST)
        path = self.path + str(_file_dir_idx).rjust(3, '0') + "/" + str(_file_idx).rjust(3, '0') + ".csv"
        # path = self.path + str(_file_dir_idx).rjust(3, '0') + "\\" + str(_file_idx).rjust(3, '0') + ".csv"
        self.data, self.label = get_one_dataset(path, self.frame)
        return self.data, self.label.long()

    def __len__(self):  # 数据集的长度
        return FILE_LEN


if __name__ == '__main__':

    RESUME = True  # 是否继续训练

    CSL_dataset = MyDataset(DATA_PATH, fps=FRAME)

    train_size = int(len(CSL_dataset) * TRAIN_SPLIT)
    test_size = len(CSL_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(CSL_dataset, [train_size, test_size])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=16, pin_memory=True, drop_last=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=16, pin_memory=True, drop_last=False)

    model = RNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYER, num_classes=NUM_CLASS,
                drop_p=DROP_OUT, batch_size=BATCH_SIZE).to(DEVICE)

    if RESUME:

        path_checkpoint = '/root/hy-tmp/best.mdl'
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint)

    model.train()
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0
    best_epoch = 0

    losses = []
    all_losses = []

    val_losses = []
    val_all_losses = []
    val_acc = []
    for e in range(EPOCH):
        # 训练
        for i, (inputs, labels) in enumerate(train_dataloader):

            inputs = Variable(inputs.to(DEVICE))
            labels = Variable(labels.to(DEVICE))
            optimizer.zero_grad()
            # 前向传播
            outputs, _ = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]

            loss = criterion(outputs, labels.squeeze())
            losses.append(loss.item())
            loss.backward(retain_graph=True)
            for p in model.parameters():
                torch.nn.utils.clip_grad.clip_grad_norm_(p, 20)  # 把梯度下降的速度控制在12以内，防止梯度爆炸
            optimizer.step()

        # 验证
        val_all_label = []
        val_all_pred = []
        with torch.no_grad():

            for j, (val_input, val_label) in enumerate(test_dataloader):
                val_input = val_input.to(DEVICE)
                val_label = val_label.to(DEVICE)
                val_out, _ = model(val_input)

                if isinstance(val_out, list):
                    val_out = val_out[0]

                val_loss = criterion(val_out, val_label.squeeze())
                val_losses.append(val_loss.item())
                prediction = torch.max(val_out, 1)[1]
                val_all_label.extend(val_label.squeeze())
                val_all_pred.extend(prediction)

        train_loss = sum(losses) / len(losses)
        all_losses.append(train_loss)

        val_loss = sum(val_losses) / len(val_losses)
        val_all_losses.append(val_loss)

        val_all_label = torch.stack(val_all_label, dim=0)
        val_all_pred = torch.stack(val_all_pred, dim=0)
        acc = accuracy_score(val_all_label.squeeze().cpu().data.squeeze().numpy(),
                             val_all_pred.cpu().data.squeeze().numpy())
        val_acc.append(acc)
        print('| epoch {:4d} | train loss {:8.5f} | val loss {:8.5f} | val accuracy {:.5f}%'
              .format(e + 1, train_loss, val_loss, acc * 100))
        if acc > best_acc:
            best_epoch = EPOCH
            best_acc = acc
            torch.save(model.state_dict(), 'best.mdl')

    epochs_range = range(EPOCH)

    fig, ax = plt.subplots(figsize=(20, 6))
    plt.rcParams['font.sans-serif'] = 'Times New Roman'

    # plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, all_losses, label='Traing Loss')
    plt.plot(epochs_range, val_all_losses, label='Validation Loss')
    plt.legend(loc='best')
    plt.title('Loss')

    # plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_acc)
    plt.title('Validation Accuracy')
    plt.show()
