import csv
import os
import matplotlib.pyplot as plt
import numpy as np 
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.models import Sequential

data_dir = "./statefulRNN"
filename = "LD2011_2014.csv"
'''
head = []
data = []
with open(os.path.join(data_dir, "LD2011_2014.txt"), "r") as f:
    flag = True 
    for line in f:
        line = line.rstrip("\n").split(";")
        if flag:
            flag = False 
            line = [x.strip("\"") for x in line]
            head = line[1:]
            continue
        for i in range(1, len(line)):
            if "," not in line[i]:
                line[i] = float(line[i])
            else:
                integer, decimal = line[i].split(",")[0], line[i].split(",")[1]
                line[i] = float(integer + "." + decimal)
        data.append(line[1:])
with open(os.path.join(data_dir, "LD2011_2014.csv"), "w", newline = "") as f:
    f_csv = csv.writer(f)
    f_csv.writerow(head)
    f_csv.writerows(data)

'''
data = []
with open(os.path.join(data_dir, filename), "r") as f:
    f_csv = csv.reader(f)
    flag = True
    for row in f_csv:
        if flag:
            flag = False
            continue
        data.append(row[1])

x_train, y_train = [], []
x_test, y_test = [], []
length = 20 
for i in range(0, len(data) // 2 - length):
    x_train.append(data[i : i+length])
    y_train.append(data[i+length])
for i in range(len(data) // 2 + 1, len(data) - length):
    x_test.append(data[i : i+length])
    y_test.append(data[i+length])

x_train = np.array(x_train)
x_train = np.expand_dims(x_train, axis = 2)
y_train = np.array(y_train)
x_test = np.array(x_test)
x_test = np.expand_dims(x_test, axis = 2)
y_test = np.array(y_test)
# print(x_train.shape, y_train.shape)
model = Sequential()
model.add(LSTM(input_dim = 1, input_length = length, units = 10, return_sequences = False))
model.add(Dense(1))
model.summary()
model.compile(optimizer = "adam", loss = "mse", metrics = ["mse", "accuracy"])

model.fit(x_train, y_train, batch_size = 96, epochs = 5, validation_split = 0.2, verbose = 1)
score = model.evaluate(x_test, y_test, verbose = 1)
print(score)