import matplotlib
matplotlib.use('TkAgg')  # 强制使用系统原生后端，完全绕开PyCharm内置插件
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import preprocessing
import random
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings .filterwarnings("ignore")


# 设置随机种子
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


def plot_grey(data):
    # 1. 核心修复：去掉多余的维度（把 (1, 76) 变成 (76,)，适配 imshow 输入）
    data = np.squeeze(data)

    # 2. 一维特征转二维图像（适配你的76维特征，重塑为8×10的灰度图，补全4个0）
    if len(data.shape) == 1:
        # 计算需要补全的长度，把76维补到80维（8×10）
        pad_length = 80 - len(data)
        # 用0填充，保证能完美reshape成8行10列
        data = np.pad(data, (0, pad_length), mode='constant', constant_values=0)
        # 重塑为二维图像
        data = data.reshape(8, 10)

    # 3. 标准绘图逻辑，完全避开PyCharm后端bug
    plt.imshow(data, cmap='gray')
    plt.axis('off')  # 隐藏坐标轴，只显示图像
    plt.show(block=True)  # 阻塞式显示，防止程序闪退，必须加！z

# t-sne初始可视化函数
def start_tsne(x_train, y_train):
    print("正在进行初始输入数据的可视化...")
    x_train1 = x_train.reshape(len(x_train), 76)
    X_tsne = TSNE().fit_transform(x_train1)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train)
    plt.colorbar()
    plt.show()

# t-sne结束可视化函数
def end_tsne(x_test, y_test, model):
    model.load_state_dict(torch.load('../model/1DCNN.pth'))
    model.eval()
    with torch.no_grad():
        hidden_features = model(torch.tensor(x_test).float()).numpy()
    X_tsne = TSNE().fit_transform(hidden_features)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test)
    plt.colorbar()
    plt.show()

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=4, stride=1, padding=1)
        self.fc1 = nn.Linear(152, 7)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        return x


# 绘制acc和loss曲线
def acc_line(history):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(len(acc))

    # 绘制accuracy曲线
    plt.plot(epochs, acc, 'r', linestyle='-.')
    plt.plot(epochs, val_acc, 'b', linestyle='dashdot')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy", "Validation Accuracy"])

    plt.figure()

    # 绘制loss曲线
    plt.plot(epochs, loss, 'r', linestyle='-.')
    plt.plot(epochs, val_loss, 'b', linestyle='dashdot')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation Loss"])
    plt.show()

# 绘制混淆矩阵
def confusion(x_test, y_test, model):
    y_pred_gailv = model(torch.tensor(x_test).float()).detach().numpy()
    y_pred_int = np.argmax(y_pred_gailv, axis=1)
    y_test_np = y_test.numpy()
    con_mat = confusion_matrix(y_test_np.astype(str), y_pred_int.astype(str))
    print(con_mat)

    plt.imshow(con_mat, cmap=plt.cm.Blues)
    indices = range(len(con_mat))

    plt.xticks(indices)
    plt.yticks(indices)

    plt.colorbar()
    plt.xlabel('guess')
    plt.ylabel('true')
    for first_index in range(len(con_mat)):
        for second_index in range(len(con_mat[first_index])):
            plt.text(first_index, second_index, con_mat[second_index][first_index], va='center', ha='center')
    plt.show()

def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val) * 255
    # normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


# 对输入到模型中的数据进一步处理
def data_pre():
    file_path = '../data/ALL_CIC-IDS-2018.xls'
    spilt_rate = 0.6

    x_train, y_train, x_valid, y_valid, x_test, y_test = preprocessing.prepro(file_path, spilt_rate)

    # 进行数据缩放0-255
    x_train = normalize_data(x_train)
    x_valid = normalize_data(x_valid)
    x_test = normalize_data(x_test)

    # 标签转为int
    y_train = [int(i) for i in y_train]
    y_valid = [int(i) for i in y_valid]
    y_test = [int(i) for i in y_test]

    # 打乱顺序
    index = [i for i in range(len(x_train))]
    random.seed(1)
    random.shuffle(index)
    x_train = np.array(x_train)[index]
    y_train = np.array(y_train)[index]

    index1 = [i for i in range(len(x_valid))]
    random.shuffle(index1)
    x_valid = np.array(x_valid)[index1]
    y_valid = np.array(y_valid)[index1]

    index2 = [i for i in range(len(x_test))]
    random.shuffle(index2)
    x_test = np.array(x_test)[index2]
    y_test = np.array(y_test)[index2]


    x_train = x_train.reshape(len(x_train), 1, 76)
    x_valid = x_valid.reshape(len(x_valid), 1, 76)
    x_test = x_test.reshape(len(x_test), 1, 76)

    return x_train, y_train, x_valid, y_valid, x_test, y_test


# main函数
def plot_gray(param):
    pass


if __name__ == '__main__':

    # 获取数据
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_pre()

    print("x_train.shape: ", x_train.shape)
    print("y_train.shape: ", y_train.shape)
    print("x_valid.shape: ", x_valid.shape)
    print("y_valid.shape: ", y_valid.shape)
    print("x_test.shape: ", x_test.shape)
    print("y_test.shape: ", y_test.shape)

    # 将标签转换为 torch.long 类型
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_valid = torch.tensor(x_valid, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_valid = torch.tensor(y_valid, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 创建训练集和验证集的 TensorDataset
    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_valid, y_valid)
    test_dataset = TensorDataset(x_test, y_test)

    # 创建 DataLoader
    batch_size = 128     # 指定批量大小
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=True)

    # 绘制灰度图
    plot_gray(x_train[0])

    # t-sne初始可视化
    start_tsne(x_train, y_train)

    # 获取定义模型
    model = MyModel()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 模型训练
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    best_accuracy = 0
    sample_num = 0  # 初始化，用于记录当前迭代中，已经计算了多少个样本
    epochs = 100

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data, target
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in validate_loader:
                data, target = data, target
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        val_loss /= len(validate_loader)
        val_acc = 100 * val_correct / val_total

        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), '../model/1DCNN.pth')

        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

    print('Finished Training')

    ################################################################# test ############################################################
    # 测试
    model.load_state_dict(torch.load('../model/1DCNN.pth'))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()

    test_acc = 100 * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')

    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(target.numpy())

    print(classification_report(y_true, y_pred, digits=4))

    # 绘制acc和loss曲线
    print("绘制acc和loss曲线")
    acc_line(history)

    # 训练结束的t-sne降维可视化
    print("训练结束的t-sne降维可视化")
    end_tsne(x_test, y_test, model)

    # 绘制混淆矩阵
    print("绘制混淆矩阵")
    confusion(x_test, y_test, model)