from keras.layers.core import Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt 
import nltk
import numpy as np 
import os 

maxlen = 0
word_freqs = collections.Counter()
num_recs = 0

with open("./LSTM/umich-si650-nlp.csv", "r", encoding = "ascii", errors = "ignore") as f:
    for line in f:
        line = line.rstrip("\n")
        sentence, label = line[:-2], line[-1]
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1

max_features = 2000
max_sentence_len = 40
vocab_size = min(max_features, len(word_freqs)) + 2
word2index = {x[0] : i + 2 for i, x in enumerate(word_freqs.most_common(max_features))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v : k for k, v in word2index.items()}

x = np.empty((num_recs, ), dtype = list)
y = np.zeros((num_recs, ))
i = 0
with open("./LSTM/umich-si650-nlp.csv", "r", encoding = "ascii", errors = "ignore") as f:
    for line in f:
        line = line.rstrip("\n")
        sentence, label = line[:-2], line[-1]
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            if word in word2index: 
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        x[i] = seqs
        y[i] = int(label)
        i += 1

'''
sequence.pad_sequneces(sequences, maxlen, dtype, padding, truncating, value)

    sequences: 列表的列表，每一个元素是一个序列。
    maxlen: 整数，所有序列的最大长度。
    dtype: 输出序列的类型。 要使用可变长度字符串填充序列，可以使用 object。
    padding: 字符串，'pre' 或 'post' ，在序列的前端补齐还是在后端补齐。
    truncating: 字符串，'pre' 或 'post' ，移除长度大于 maxlen 的序列的值，要么在序列前端截断，要么在后端。
    value: 浮点数，表示用来补齐的值。
'''
x = sequence.pad_sequences(x, maxlen = max_sentence_len, padding = "post", truncating = "post", value = 0)
# 划分训练集和测试集
'''
sklearn.model_selection.train_test_split(*arrays, **options)
Parameters:	

*arrays : sequence of indexables with same length / shape[0]

    Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
test_size : float, int or None, optional (default=None)

    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
train_size : float, int, or None, (default=None)

    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.
random_state : int, RandomState instance or None, optional (default=None)

    If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
shuffle : boolean, optional (default=True)

    Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
stratify : array-like or None (default=None)

    If not None, data is split in a stratified fashion, using this as the class labels.

Returns:	
splitting : list, length=2 * len(arrays)

    List containing train-test split of inputs.

    New in version 0.16: If the input is sparse, the output will be a scipy.sparse.csr_matrix. Else, output type is the same as the input type.
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# LSTM超参数
embedding_size = 128
hidden_layer_size = 64
batch_size = 32
num_epoch = 10

model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length = max_sentence_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(hidden_layer_size, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(1, activation = "sigmoid"))
model.summary()

model.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ["accuracy"])
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = num_epoch, validation_split = 0.2, verbose = 1)
score = model.evaluate(x_test, y_test, verbose = 1)
print("test score:", score[0])
print("test accuracy:", score[1]) 

# 绘制训练过程中训练集上的准确率和验证集上的正确率以及对应的损失值
plt.figure()

plt.subplot(121)
plt.title("Accuracy")
plt.plot(history.history["acc"], color = "r", label = "train")
plt.plot(history.history["val_acc"], color = "g", label = "validation")
plt.legend(loc = "best")

plt.subplot(122)
plt.title("Loss")
plt.plot(history.history["loss"], color = "r", label = "train")
plt.plot(history.history["val_loss"], color = "g", label = "valdation")
plt.legend(loc = "best")

plt.show()