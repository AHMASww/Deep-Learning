from keras.layers.core import Reshape
from keras.layers import merge, Input, Lambda, Dense
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from keras.preprocessing.text import *
from keras.preprocessing.sequence import skipgrams
import keras.backend as K
import tensorflow as tf

from sklearn.model_selection import train_test_split

import collections
import os 
import nltk 
import numpy as np
import matplotlib.pyplot as plt

data_dir = "./Alice"
filename = "28885.txt"

data = []
# word_freq = collections.defaultdict(int)

with open(os.path.join(data_dir, filename), "r", encoding = "ascii", errors = "ignore") as f:
    for line in f:
        line = line.rstrip("\n").lower()
        if len(line) == 0:
            continue
        data.append(line)

data_str = "".join(data)

# 将原文按句子划分
sents = nltk.sent_tokenize(data_str)
# Tokenizer()函数将文本生产一个token序列
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sents)
# 建立单词和ID的映射表
word2index = tokenizer.word_index
word2index[0] = "UNK"
index2word = {value : key for key, value in word2index.items()}
# 构造训练集和测试集
wids = [word2index[w] for w in text_to_word_sequence(data_str)]
# 使用skipgrams函数对文本进行采样，window_size是半个窗口，即[center_word-window_size, center_word+window_size+1]
pairs, labels = skipgrams(wids, vocabulary_size = len(word2index), window_size = 1, negative_samples = 1)
pairs, labels = np.array(pairs), np.array(labels)
x_train, x_test, y_train, y_test = train_test_split(pairs, labels, test_size = 0.3)
# 搭建模型
voca_size = len(word2index)
embed_size = 300

input_word = Input(shape = (2,), name = "input_1")
embedding = Embedding(input_dim = voca_size, output_dim = embed_size, 
                      input_length = 2, embeddings_initializer = "glorot_uniform", name = "embedding_1")(input_word)
lambda_dot = Lambda(lambda x : K.prod(x, axis = 1), output_shape = (embed_size,))(embedding)
dense1 = Dense(units = 1, activation = "sigmoid", name = "dense_1")(lambda_dot)

model = Model(inputs = input_word, outputs = dense1)
model.summary()
model.compile(optimizer = "adam", loss = "mse")

# 训练模型
history = model.fit(x_train, y_train, batch_size = 32, epochs = 5, verbose = 1, validation_split = 0.2)

# 测试模型
model.evaluate(x_test, y_test, verbose = 1)

# 绘制结果
loss = history.history["loss"]
val_loss = history.history["val_loss"]
times = [i for i in range(len(loss))]

plt.figure()
plt.plot(times, loss, c = "r", label = "Training Loss")
plt.plot(times, val_loss, c = "b", label = "Validation Loss")
plt.show()