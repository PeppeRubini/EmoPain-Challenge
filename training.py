import numpy as np
import pandas as pd
import os
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense


def csv_to_array(path_geo, path_label):
    df_label = pd.DataFrame()
    df_geo = pd.DataFrame()
    for file in os.listdir(path_label):
        df_label = pd.concat([df_label, pd.read_csv(path_label + file, header=None)])
        df_geo = pd.concat([df_geo, pd.read_csv(path_geo + file).dropna()])
    df_label = df_label.iloc[:, [0]]
    arr_label = df_label.to_numpy().astype(np.int64)
    arr_geo = df_geo.to_numpy()

    return arr_geo, arr_label


def array_input_step(x, step):
    X = []
    i = 0
    for e in x[:-(step-1)]:
        l = []
        l.append(list(e))
        i += 1
        for j in range(i,step + i - 1):
            l.append(list(x[j]))
        X.append(l)
    return np.array(X)


def array_output_step(y, step):
    Y = y[step-1:]
    return np.array(Y)


path_geo_train = "./CleanDataset/Face Features/Geometric Features/train/"
path_label_train = "./CleanDataset/Pain Labels/train/"
path_geo_valid = "./CleanDataset/Face Features/Geometric Features/valid/"
path_label_valid = "./CleanDataset/Pain Labels/valid/"
time_step = 3
num_features = 17
x_train, y_train = csv_to_array(path_geo_train, path_label_train)
x_valid, y_valid = csv_to_array(path_geo_valid, path_label_valid)
X_t = array_input_step(x_train, time_step)
Y_t = array_output_step(y_train, time_step)
X_v = array_input_step(x_valid, time_step)
Y_v = array_output_step(y_valid, time_step)

model = Sequential()
model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.2,
               input_shape=(time_step, num_features), return_sequences=True))
model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.2))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])

model.fit(X_t, Y_t, epochs=10)

X_v = X_v.reshape((len(X_v), time_step, num_features))

y_test = model.predict(X_v)
print(y_test)
