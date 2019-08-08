from keras.layers import Dense, Embedding, Input, Lambda, Activation
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer 
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras.backend as K
import numpy as np 
import nltk 
import os 

data_dir = "./Alice"
filename = "28885.txt"

data = []
with open(os.path.join(data_dir, filename), "r", encoding = "ascii", errors = "ignore") as f:
    for line in f:
        line = line.strip("").lower()
        line = line.rstrip("\n")
        if len(line) < 10:
            continue
        data.append(line)

data_str = "".join(data)

# 以句子为单位划分语料
sents = nltk.sent_tokenize(data_str)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sents)
word2index = tokenizer.word_index
index2word = {value : key for key, value in word2index.items()}
sequences = tokenizer.texts_to_sequences(sents)

# 构建训练集、测试集
x, y = [], []
for sequence in sequences:
    triples = list(nltk.trigrams(sequence))
    for item in triples:
        x.append([item[0], item[2]])
        l = to_categorical(item[1], num_classes = len(word2index)+1, dtype = "int32")
        y.append(l)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# 搭建模型
voca_size = len(word2index) + 1
embed_size = 300

model = Sequential()
model.add(Embedding(input_dim = voca_size, output_dim = embed_size, input_length = 2, embeddings_initializer = "glorot_uniform"))
model.add(Lambda(lambda x : K.mean(x, axis = 1), output_shape = (embed_size,)))
# model.add(Activation("relu"))
model.add(Dense(voca_size, kernel_initializer = "glorot_uniform", activation = "softmax"))
model.summary()

# 编译模型
model.compile(loss = "categorical_crossentropy", optimizer = "adam")

# 训练模型
early_stopping = EarlyStopping(monitor = "val_loss", patience = 2, verbose = 1)
model.fit(x_train, y_train, batch_size = 32, epochs = 30, verbose = 1, validation_split = 0.1, callbacks = [early_stopping])

#测试模型
score = model.evaluate(x_test, y_test, verbose = 1)
print(score)