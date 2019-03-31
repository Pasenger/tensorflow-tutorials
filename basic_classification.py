# -*-conding:utf-8 -*-

# =====================================
# 基本分类
# https://tensorflow.google.cn/tutorials/keras/basic_classification
# =====================================

# Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# 导入Fashion MNIST数据集， 下载目录在：C:\Users\Pasenger\.keras\datasets\fashion-mnist
fashion_mnist = keras.datasets.fashion_mnist

# 加载数据集会返回4个Numpy数组
# 训练集：train_images, train_labels
# 测试集：test_images, test_labels
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 每张图片都映射到一个标签
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
# (60000, 28, 28)

print(test_images.shape)
# (10000, 28, 28)

# 预处理数据
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 图像：长：28， 宽：28， 高：255
# 我们将这些值缩小到 0 到 1 之间，然后将其馈送到神经网络模型。为此，将图像组件的数据类型从整数转换为浮点数，然后除以 255。以下是预处理图像的函数：
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()

# 构建模型

# 设置层
# 神经网络的基本构造块是层。层从馈送到其中的数据中提取表示结果。希望这些表示结果有助于解决手头问题。
# 大部分深度学习都会把简单层连接在一起，大部分层都具有在训练期间要学习的参数。

model = keras.Sequential([
    # 第一层： 将图像格式从二维数组(28 * 28像素)转换成一维数组(28 * 28 = 784像素)
    # 可以将该层视为图像中像素未堆叠的行，并排列这些行。
    # 该层没有需要学习的参数，只改动数据的格式。完成扁平化
    keras.layers.Flatten(input_shape=(28, 28)),

    # 第一个Dense层：具有128个节点（或神经元）
    keras.layers.Dense(128, activation=tf.nn.relu),

    # 第二个Dense层：具有10个节点的softmax层，返回一个具有10个概率得分的数组，这些得分总和为1
    # 用来表示当前图像属于10个类别中某一个的概率
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 编译模型
# 模型还需要进行迹象设置才可以开始训练：
# 损失函数：衡量模型在训练期间的准确率。
# 优化器：根据模型看到的数据及其损失函数更新模型的方式
# 指标：用于监控训练和测试步骤
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
# 训练模型需要执行以下步骤：
# 1. 将训练数据馈送到模型中，本例中为train_iamges和tarin_labels数组
# 2. 模型学习将图像与标签相关联
# 3. 要求模型对测试集进行预测，本例中为test_iamges数据，会验证预测结果是否与test_labels数组中的标签一致

# 开始训练，请调用model.fit方法，使模型与训练数据“拟合”
model.fit(train_images, train_labels, epochs=5)

# 评估准确率
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: ', test_acc)

# 结果表明，模型在测试数据集上的准确率略低于在训练数据集上的准确率。
# 训练准确率和测试准确率之间的这种差异表示出现过拟合。
# 如果机器学习模型在新数据上的表现不如在训练数据上的表现，就表示出现过拟合。


# 做出预测

# 模型经过训练后，可以使用它对一些图像进行预测
predictions = model.predict(test_images)

# 查看第一个预测结果
print(predictions[0])

# 查看可可信度最大的标签
print('预测：', np.argmax(predictions[0]))
print('实际：', test_labels[0])


# 将预测结果绘制成图来查看全部10个通道
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("P:{} {:2.0f}% (T:{})".format(
        class_names[predicted_label],
        100 * np.max(predictions_array),
        class_names[true_label]
    ), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# 第0张
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)

plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)

plt.show()


# 使用经过训练的模型对单个图像进行预测

# tf.keras 模型已经过优化，可以一次性对样本批次或样本集进行预测。因此，即使我们使用单个图像，仍需要将其添加到列表中：
img = test_images[0]
img = (np.expand_dims(img, 0))

predictions_single = model.predict(img)
print('预测结果： ', predictions_single)
print('预测值：', np.argmax(predictions_single[0]))

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
