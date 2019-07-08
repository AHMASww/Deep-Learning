from __future__ import print_function, division
import numpy as np 
from keras.datasets import mnist
from keras.models import Sequential, save_model
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import os
np.random.seed(1671)

DIR = r".\tmp"

# 数据，混合并划分训练集和测试集数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
# 归一化
x_train /= 255
x_test /= 25
# 将类向量转换为二值类别矩阵
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
# 构建模型
model = Sequential()

model.add(Dense(128, input_shape = (784,)))
model.add(Activation("relu"))
model.add(Dropout(0.3))

model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.3))

model.add(Dense(10))
model.add(Activation("softmax"))

model.summary()
# 编译模型
model.compile(loss = "categorical_crossentropy", optimizer = Adam(), metrics = ["accuracy"])
# 保存最好的模型
checkpoint = ModelCheckpoint(filepath = os.path.join(DIR, "model-{epoch:02d}.h5"))
# 训练模型
history = model.fit(x_train, y_train, batch_size = 128, epochs = 10, verbose = 1, validation_split = 0.2, callbacks = [checkpoint])
# 测试模型
score = model.evaluate(x_test, y_test, verbose = 1)
print("Test score:", score[0])
print("Test accuracy:", score[1])