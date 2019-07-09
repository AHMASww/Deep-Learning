from keras import backend
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.datasets import mnist 
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np 
import matplotlib.pyplot as plt 

# 定义ConvNet
class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        # Conv ==> Relu ==> Pool
        model.add(Conv2D(filters = 20, kernel_size = 5, padding = "same", input_shape = input_shape, activation = "relu"))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        
        #Conv ==> Relu ==> Pool
        model.add(Conv2D(filters = 50, kernel_size = 5, border_mode = "same", activation = "relu"))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

        # Flatten层到relu层
        model.add(Flatten())
        model.add(Dense(500, activation = "relu"))

        #softmax分类器
        model.add(Dense(classes, activation = "softmax"))

        return model 

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 后端采用tensorflow，采用tensorflow的图像表示顺序，与theano有区别
backend.set_image_dim_ordering("tf")

# 数据归一化
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

# mnist数据是二维的，这里输入要求是三维的，改变数据的结构
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]

# 将类向量转换成二值类别矩阵
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# 初始化优化器和模型
model = LeNet.build((28, 28, 1), 10)
model.compile(optimizer = Adam(), loss = "categorical_crossentropy", metrics = ["accuracy"])
history = model.fit(x_train, y_train, batch_size = 128, epochs = 20, verbose = 1, validation_split = 0.2)
score = model.evaluate(x_test, y_test, verbose = 1)
print("Test socre:", score[0])
print("Test accuracy:", score[1])