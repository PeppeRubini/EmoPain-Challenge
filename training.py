from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, AvgPool2D, Dense, LSTM
import os
import pandas as pd
import numpy as np

path_geo_train = "./CleanDataset/Face Features/Geometric Features/train/"
path_label_train = "./CleanDataset/Pain Labels/train/"
path_geo_valid = "./CleanDataset/Face Features/Geometric Features/valid/"
path_label_valid = "./CleanDataset/Pain Labels/valid/"


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


x_train, y_train = csv_to_array(path_geo_train, path_label_train)
x_valid, y_valid = csv_to_array(path_geo_valid, path_label_valid)

"""print(x_valid.shape)
print(y_valid.shape)
print(x_train[0])
print(x_train[x_train.shape[0]-1])
print("............................")
print(y_train[0])
print(y_train[y_train.shape[0]-1])
print("-----------------------")
print(x_valid[0])
print(x_valid[x_valid.shape[0]-1])
print("............................")
print(y_valid[0])
print(y_valid[y_valid.shape[0]-1])"""

model = Sequential()
# model.add(Conv2D(units=64, activation='relu'))
# model.add(MaxPool2D(units=32))
"""model.add(LSTM(units=64, activation='relu'))
model.add(LSTM(units=32, activation='relu'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])"""
time_steps = 1
num_features = 39
x_train = x_train.reshape(x_train.shape[0], time_steps, num_features)
model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.2,
               input_shape=(time_steps, num_features), return_sequences=True))
model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.2))
model.add(Dense(1))

"""model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))"""

model.compile(optimizer="adam", loss="mse", metrics="accuracy")
model.fit(x_train, y_train, epochs=5, batch_size=32)
loss_and_metrics = model.evaluate(x_valid, y_valid, batch_size=128)
