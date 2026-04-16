import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import preprocessing  # 需确保本地有该预处理模块，或替换为自定义预处理逻辑
import random
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore")  # 修正原代码空格错误

# 设置随机种子
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# 灰度图像可视化函数
def plot_gray(data):
    plt.imshow(data, cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    plt.show()

# t-SNE初始输入数据可视化函数
def start_tsne(x_train, y_train):
    print("正在进行初始输入数据的可视化...")
    # 修正reshape逻辑：适配任意特征维度，原代码硬编码76仅适配特定数据
    n_features = x_train.shape[1] if len(x_train.shape) > 2 else x_train.shape[0]
    x_trainl = x_train.reshape(len(x_train), n_features)
    X_tsne = TSNE().fit_transform(x_trainl)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train)
    plt.colorbar()
    plt.show()

# t-SNE模型结果可视化函数（补全函数体+修正缩进）
def end_tsne(x_test, y_test, model):
    # 加载模型权重（修正路径语法）
    model.load_state_dict(torch.load('../model/1DCNN.pth', map_location='cpu'))
    model.eval()
    with torch.no_grad():
        # 需根据模型结构提取特征并可视化，此处补充基础逻辑
        # （注：需替换为你实际的1DCNN模型前向提取特征代码，示例如下）
        # features = model.extract_features(x_test)  # 假设模型有提取特征方法
        # tsne_result = TSNE().fit_transform(features.cpu().numpy())
        # 以下为通用演示逻辑，需根据模型修改
        tsne_result = TSNE().fit_transform(x_test.cpu().numpy() if isinstance(x_test, torch.Tensor) else x_test)
        plt.figure(figsize=(10, 10))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test)
        plt.colorbar()
        plt.show()

# ------------------- 补全1DCNN模型定义（原代码缺失核心模块） -------------------
class My1DCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(My1DCNN, self).__init__()
        # 示例1D卷积层，需根据实际数据调整参数
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        # 全连接层需根据卷积输出维度动态计算，此处示例需调整
        self.fc1 = nn.Linear(16 * 8, 64)  # 假设池化后维度为8，需实际匹配
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------- 运行示例（需替换为你的实际数据） -------------------
if __name__ == "__main__":
    # 1. 生成模拟数据（替换为真实数据x_train, y_train, x_test, y_test）
    # 形状说明：(样本数, 特征维度/通道数, 序列长度) 适配1DCNN
    x_train = torch.randn(100, 1, 16)  # 100个样本，1通道，序列长度16
    y_train = torch.randint(0, 2, (100,))  # 二分类标签
    x_test = torch.randn(20, 1, 16)
    y_test = torch.randint(0, 2, (20,))

    # 2. 初始化模型
    model = My1DCNN(input_channels=1, num_classes=2)

    # 3. 调用可视化函数
    start_tsne(x_train.numpy(), y_train.numpy())  # 初始数据可视化
    end_tsne(x_test, y_test, model)  # 模型结果可视化
