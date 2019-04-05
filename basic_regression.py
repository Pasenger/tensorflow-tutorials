# -*-conding:utf-8 -*-

# =======================================================================
# Regression: predict fuel efficiency
# https://tensorflow.google.cn/tutorials/keras/basic_regression#the_auto_mpg_dataset
# =======================================================================

# ========================================================================
# 总结：
# 均方误差(MSE)是回归问题中常用的损失函数(分类问题使用不同的损失函数)。
# 类似地，用于回归的评估指标与分类不同。一个常见的回归度量是平均绝对误差(MAE)。
# 当数值输入数据特性具有不同范围的值时，应该将每个特性独立地缩放到相同的范围。
# 如果没有太多的训练数据，一种技术是选择一个具有很少隐藏层的小网络，以避免过度拟合。
# 早期停止是一种有效的技术，以防止过度拟合。
# ========================================================================

from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# get daata
dataset_path = keras.utils.get_file(
    "auto-mpg.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)

print(dataset_path)

column_names = ['MPG',              # 每加仑行驶的英里数
                'Cylinders',        # 气缸
                'Displacement',     # 排量
                'Horsepower',       # 马力
                'Weight',           # 重量
                'Acceleration',     # 加速
                'Model Year',       # 车型年代
                'Origin'            # 来源
                ]

raw_dataset = pd.read_csv(dataset_path,
                          names=column_names,
                          na_values="?",
                          comment='\t',
                          sep=" ",
                          skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

# clean data
# print(dataset.isna().sum())
dataset = dataset.dropna()

# "Origin"列是类型，转化为One-hot
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
print(dataset.tail())

# 将数据分为训练集和测试集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 检查数据
sns.pairplot(train_dataset[
    [
        'MPG', 'Cylinders', 'Displacement', 'Weight'
    ]
], diag_kind='kde')
plt.show()

train_stats = train_dataset.describe()
train_stats.pop('MPG')
train_stats = train_stats.transpose()
print(train_stats)

# 从标签中分离特性
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# 标准化数据
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# 构建模型
def build_model():
    _model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = keras.optimizers.RMSprop(0.001)

    _model.compile(
        loss='mean_squared_error',      # 均方误差(MSE)是回归问题中常用的损失函数(分类问题使用不同的损失函数)。
        optimizer=optimizer,
        metrics=['mean_absolute_error', 'mean_squared_error']
    )

    return _model


model = build_model()
print(model.summary())

# 尝试模型
# example_batch = normed_train_data[:10]
# example_result = model.predict(example_batch)
#
# print(example_result)


# 训练模型
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print(' ')
        print('.', end='')


EPOCHS = 1000

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10
)

_history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop, PrintDot()]
)


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(
        hist['epoch'],
        hist['mean_absolute_error'],
        label='Train Error'
    )
    plt.plot(
        hist['epoch'],
        hist['val_mean_absolute_error'],
        label='Val Error'
    )
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(
        hist['epoch'],
        hist['mean_squared_error'],
        label='Train Error'
    )
    plt.plot(
        hist['epoch'],
        hist['val_mean_squared_error'],
        label='Val Error'
    )
    plt.ylim([0, 20])
    plt.legend()

    plt.show()


plot_history(_history)


# 做出预测
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()
