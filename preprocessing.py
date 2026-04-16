# github: https://github.com/boating-in-autumn-rain?tab=repositories
# 抖音: lilian
# 咨询微信: gouhuang12
from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from sklearn.model_selection import train_test_split

# 数据预处理代码
def prepro(file_path, spilt_rate):
    # 读取xls文件
    df = pd.read_excel(file_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.fillna(0, inplace=True)

    # 假设最后一列是标签
    X = df.iloc[1:, :-1]  # 所有行，除了最后一列的数据
    y = df.iloc[1:, -1]  # 所有行的最后一列数据

    # 划分数据集
    # 先将数据分为训练集和剩余部分（测试集+验证集）
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1-spilt_rate, random_state=42)

    # 再将剩余部分划分为测试集和验证集
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)

    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)


    # 输出数据集的大小以确认划分
    # print("x_train.shape: ", X_train.shape)
    # print("y_train.shape: ", y_train.shape)
    # print("x_valid.shape: ", X_val.shape)
    # print("y_valid.shape: ", y_val.shape)
    # print("x_test.shape: ", X_test.shape)
    # print("y_test.shape: ", y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

#
# if __name__ == '__main__':
#     file_path = '../data/ALL_CIC-IDS-2018.xls'
#     spilt_rate = 0.6
#
#     X_train, y_train, X_val, y_val, X_test, y_test = prepro(file_path, spilt_rate)
