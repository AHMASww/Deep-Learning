from keras import backend
from keras.datasets import cifar10
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
import numpy as np 

# 分类种类
classes = 10

# 定义网络结构
class CNN:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        # conv ==> conv == > relu ==> pool
        model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", input_shape = input_shape, activation = "relu"))
        model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu"))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        model.add(Dropout(0.25))

        # conv ==> conv ==> relut ==>pool
        model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu"))
        model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu"))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        model.add(Dropout(0.25))

        # Flatten层到Dense
        model.add(Flatten())
        model.add(Dense(512, activation = "relu"))
        model.add(Dropout(0.25))
        model.add(Dense(classes, activation = "softmax"))

        return model


# 加载数据
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train[0].shape)

# 数据归一化
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

# 将类别矩阵转为二值分类矩阵
# CIFAR-10数据中有十种分类：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、轮船和卡车
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# 选用tensorflow作为后端 
backend.set_image_dim_ordering("tf")

# 构建模型
model = CNN.build(input_shape = x_train[0].shape, classes = classes)
# 模型编译
model.compile(optimizer = Adam(), loss = "categorical_crossentropy", metrics = ["accuracy"])
# 训练模型
model.fit(x_train, y_train, batch_size = 128, epochs = 40, verbose = 1, validation_split = 0.2)
# 评估模型
socre = model.evaluate(x_test, y_test, verbose = 1)
print("test socre:", socre[0])
print("test accuracy:", socre[1])
