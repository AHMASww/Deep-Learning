from keras.layers.core import Dense, Dropout, Activation, RepeatVector, SpatialDropout1D
from keras.layers import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import collections
import nltk
import numpy as np 
import os 
import csv

'''
data_dir = "./GRU"
# 读取宾州树库的样本
sents = nltk.corpus.treebank.tagged_sents()
sent_words, sent_poss = [], []
for sent in sents:
    words, poss = [], []
    for word, pos in sent:
        if pos == "-None-": continue
        words.append(word)
        poss.append(pos)
    sent_words.append(words)
    sent_poss.append(poss) 
# 将数据写入到csv文件中
with open(os.path.join(data_dir, "words.csv"), "w", newline = "") as f:
    f_csv = csv.writer(f)
    f_csv.writerows(sent_words)
with open(os.path.join(data_dir, "poss.csv"), "w", newline = "") as f:
    f_csv = csv.writer(f)
    f_csv.writerows(sent_poss)
'''
'''
# 通过这个函数来浏览一次数据，以便得知该数据集中有多少个句子、词汇量、标注等
def parse_sentences(word_filename, poss_filename):
    word_freqs = collections.Counter()
    poss = set()
    num_recs, max_len = 0, 0
    with open(word_filename, "r", encoding = "ascii") as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            for word in row:
                word = word.rstrip().lower()
                word_freqs[word] += 1
            if len(row) > max_len:
                max_len = len(row)
            num_recs += 1

    with open(poss_filename, "r", encoding = "ascii") as f:
        f_csv = csv.reader(f)
        for row in f:
            for pos in row:
                poss.add(pos)
    
    return(num_recs, max_len, word_freqs, poss)
'''

max_sentence_len = 272
word_max_freq = 5000
poss_max_freq = 33
# 构建对应的映射列表
word_freq = collections.Counter()
with open("./GRU/words.csv", "r", encoding = "ascii") as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        for word in row:
            word_freq[word] += 1
word2index = {x : i+2 for i, x in enumerate(word_freq.most_common(word_max_freq))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {value : key for key, value in word2index.items()}

poss_freq = collections.Counter()
with open("./GRU/poss.csv", "r", encoding = "ascii") as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        for pos in row:
            poss_freq[pos] += 1
poss2index = {x : i+1 for i, x in enumerate(poss_freq)}
poss2index["PAD"] = 0
index2poss = {value : key for key, value in poss2index.items()}

# 将输入属性转为数字表示，标签转为one-hot表示
x, y = [], []
with open("./GRU/words.csv", "r", encoding = "ascii", errors = "ignore") as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        temp_x = []
        for word in row:
            if word in word2index:
                temp_x.append(word2index[word])
            else:
                temp_x.append(word2index["UNK"])
        x.append(temp_x)

with open("./GRU/poss.csv", "r", encoding = "ascii", errors = "ignore") as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        temp_y = []
        for pos in row:
            temp_y.append(poss2index[pos])
        y.append(temp_y)

x = sequence.pad_sequences(x, maxlen = max_sentence_len, value = 0, padding = "post")
y = sequence.pad_sequences(y, maxlen = max_sentence_len, value = 0, padding = "post")
y = np_utils.to_categorical(y, num_classes = len(poss2index))

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# 超参数
embed_size = 128
hidden_size = 64
batch_size = 32
num_epochs = 1

# 模型搭建
model = Sequential()
model.add(Embedding(input_dim = len(word2index), output_dim = embed_size, input_length = max_sentence_len))
model.add(SpatialDropout1D(0.2))
model.add(GRU(hidden_size, dropout = 0.2, recurrent_dropout = 0.2, return_sequences = False))
model.add(RepeatVector(max_sentence_len))
model.add(GRU(hidden_size, dropout = 0.2, recurrent_dropout = 0.2, return_sequences = True))
model.add(TimeDistributed(Dense(len(poss2index), activation = "softmax")))

#
model.summary()

# 模型编译
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

# 训练模型
model.fit(x_train, y_train, batch_size = batch_size, epochs = num_epochs, validation_split = 0.2, verbose = 1)
# 评估模型
score = model.evaluate(x_test, y_test)
print("test score:", score[0])
print("test accuracy:", score[1])