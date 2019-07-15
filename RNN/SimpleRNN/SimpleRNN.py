from __future__ import print_function
from keras.layers import Dense
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
import numpy as np 

def dataProcess(fileName, length, step):
    # 获取原始数据
    lines = []

    with open(file = fileName, mode = "r", encoding = "ascii", errors = "ignore") as f:
        for line in f:
            line = line.strip().lower()
            # line = line.decode("ascii", "ignore")
            if len(line) == 0:
                continue
            lines.append(line)
    
    text = " ".join(lines)

    # 进一步处理数据
    # 建立查存表
    chars = set([c for c in text])
    nb_chars = len(chars)
    char2index = dict((c, i) for i, c in enumerate(chars))
    index2char = dict((i, c) for i, c in enumerate(chars))

    # 创建输入和标签文本
    input_chars = []
    input_chars_label = []
    for i in range(0, len(text) - length, step):
        input_chars.append(text[i : i+length])
        input_chars_label.append(text[i+length])

    # 将输入和标签文本向量化
    x = np.zeros((len(input_chars), length, nb_chars), dtype = np.bool)
    y = np.zeros((len(input_chars), nb_chars), dtype = np.bool)
    for i, input_char in enumerate(input_chars):
        for j, ch in enumerate(input_char):
            x[i, j, char2index[ch]] = 1
        y[i, char2index[input_chars_label[i]]] = 1

    return(nb_chars, char2index, index2char, input_chars, input_chars_label, x, y)

def rnn(nb_chars, length, input_chars, nums_interation, char2index, index2char):
    # 构建模型
    model = Sequential()
    model.add(SimpleRNN(units = 128, return_sequences = False, input_shape = (length, nb_chars), unroll = True))
    model.add(Dense(nb_chars, activation = "softmax"))
    # 编译模型
    model.compile(loss = "categorical_crossentropy", optimizer = "rmsprop")

    # 训练模型
    for iteration in range(nums_interation):
        print("=" * 50)
        print("Iteration :", iteration)
        model.fit(x, y, batch_size = 128, epochs = 1)

        test_idx = np.random.randint(len(input_chars))
        test_chars = input_chars[test_idx]
        print("Generating from seed :", test_chars)
        print(test_chars, end = "")
        for t in range(100):
            xtest = np.zeros((1, length, nb_chars), dtype = np.bool)
            for i, ch in enumerate(test_chars):
                xtest[0, i, char2index[ch]] = 1
            pred = model.predict(xtest, verbose = 0)
            ypred = index2char[np.argmax(pred)]
            print(ypred, end = "")
            # 使用test_chars + ypred继续作为输入
            test_chars = test_chars[1:] + ypred
        # 换行
        print()


if __name__ == "__main__":
    fileName = "./Alice/28885.txt"
    # 获取数据
    nb_chars, char2index, index2char, input_chars, input_chars_label, x, y = dataProcess(fileName, length = 30, step = 1)
    # rnn模型建立训练测试
    rnn(nb_chars, 30, input_chars, 50, char2index, index2char)