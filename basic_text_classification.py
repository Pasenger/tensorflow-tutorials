# -*-conding:utf-8 -*-

# =======================================================================
# 影评文本分类
# https://tensorflow.google.cn/tutorials/keras/basic_text_classification
# =======================================================================

import tensorflow as tf
from tensorflow import keras

# import numpy as np
import matplotlib.pyplot as plt

# 下载IMDB数据集
imdb = keras.datasets.imdb

# 参数 num_words=10000 会保留训练数据中出现频次在前 10000 位的字词。为确保数据规模处于可管理的水平，罕见字词将被舍弃。
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 探索数据
print('Training entries: {}, labels: {}'.format(len(train_data), len(train_labels)))

# 影评文本已转换为整数，其中每个整数都表示字典中的一个特定字词。
print(train_data[0])

# 影评的长度可能会不同
print(len(train_data[0]), len(train_data[1]))

# 将整数转换回字词
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
# unknown
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(val, key) for (key, val) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(train_data[0]))

# ==========准备数据===============
# 影评数据(整数数组)必须转换为张量，然后才能馈送到神经网络中。我们可以通过以下两种方式进行转换：
# 1. 对数组进行独热编码
#    将他们转换成由0和1构成的向量。如，序列[3, 5]将变成一个10000维度的向量，除索引3和5转换为1之外，
#    其余的全部转换为0。 然后，将它作为网络的第一层，一个可以处理浮点向量数据的密集层。不过，这种方法
#    会占用大量内存，需要一个大小为num_words * num_reviews的矩阵。
# 2. 填充数组，使它们都具有相同的长度，然后创建一个形状为max_length * num_reviews的整数张量。
#    可以使用过一个能够处理这种形状的嵌入层作为网络中的第一层
#
# 这里使用第二种
# ==========准备数据===============

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)

print('after preprocessing: train_data.len: {}'.format(len(train_data[0])))
print(train_data[0])


# ==========构建模型===============
# 神经网络通过堆叠层创建而成，需要做出两个架构方面的主要决策：
# 1. 要在模型中使用多少个层？
# 2. 要针对每个层使用多个个隐藏单元?
# ==========构建模型===============

# 输入数据由字词-索引数组构成。要预测的标签是0或1
vocab_size = 10000
model = keras.Sequential()

# 第一层是Embedding层，会在整数编码的词汇表中查找每个单词-索引的嵌入向量。
# 模型在接受训练时会学习这些向量。这些向量会像输出数组添加一个维度，
# 生成维度为：(batch, sequence, embedding)
model.add(keras.layers.Embedding(vocab_size, 16))

# 第二层使用一个GlobalAveragePooling1D层通过对序列维度求平均值，针对米格样本
# 返回一个长度固定的输出向量，这样，模型便能够以尽可能简单的方式处理各种输入长度。
model.add(keras.layers.GlobalAveragePooling1D())

# 第三层：上一层的输出向量会出入一个全连接(Dense)层，包含16个隐藏单元
model.add(keras.layers.Dense(16, activation=tf.nn.relu))

# 加入dropout解决过拟合
model.add(keras.layers.Dropout(0.5))

# 第四层：与单个输出节点密集连接。应用sigmoid激活函数后，结果时介于0到1之间的浮点数，
# 表示概率或置信水平
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

print(model.summary())

# 损失函数和优化器
# 模型在训练时需要一个损失函数和一个优化器。
# 由于这是一个二元分类问题且模型会输出一个概率（应用 S 型激活函数的单个单元层），
# 因此我们将使用 binary_crossentropy 损失函数。
# 该函数并不是唯一的损失函数，例如，您可以选择 mean_squared_error。
# 但一般来说，binary_crossentropy 更适合处理概率问题，它可测量概率分布之间的“差距”，
# 在本例中则为实际分布和预测之间的“差距”。
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 创建验证集
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 训练模型
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

# 评估模型
results = model.evaluate(test_data, test_labels)
print('evaluate result: ', results)


# 创建准确率和损失随时间变化的图
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history.history['acc']
val_acc_values = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()



