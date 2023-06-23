import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense


def csv_to_array(path_dataset):
    df_label = pd.DataFrame()
    df_geo = pd.DataFrame()
    for file in os.listdir(path_dataset):
        table = pd.read_csv(path_dataset + file)
        df_label = pd.concat([df_label, table.pop("pain_label")])
        df_geo = pd.concat([df_geo, table])

    return df_geo.to_numpy(), df_label.to_numpy().astype(np.int64)


def array_stepping(x, y, step, overlapping_rate):
    X = []
    Y = []
    overlapping = int(step * overlapping_rate)
    start = 0
    end = start + step
    while end < len(x):
        # print(f"{start}:{end}=>{x[start:end]}>>>>{y[end-2]}")
        l = x[start:end]
        start = end - overlapping
        end = start + step
        X.append(l)
        Y.append(y[end - overlapping])
    return np.array(X), np.array(Y)


path_clean_dataset_train = "./CleanDataset/train/"
path_clean_dataset_valid = "./CleanDataset/valid/"
time_step = 4
num_features = 17
overlapping_rate = 0.5
x_train, y_train = csv_to_array(path_clean_dataset_train)
x_valid, y_valid = csv_to_array(path_clean_dataset_valid)

"""import utils
dict_train = utils.label_dict(y_train)
dict_valid = utils.label_dict(y_valid)
print("\ntraining")
utils.print_dict(dict_train)
print("\nvalidation")
utils.print_dict(dict_valid)
utils.pie_chart_lable(utils.label_dict(y_train))
utils.pie_chart_lable(utils.label_dict(y_valid))"""

X_t, Y_t = array_stepping(x_train, y_train, time_step, overlapping_rate)
X_v, Y_v = array_stepping(x_valid, y_valid, time_step, overlapping_rate)
"""dict_train = utils.label_dict(Y_t)
dict_valid = utils.label_dict(Y_v)
print("\ntraining")
utils.print_dict(dict_train)
print("\nvalidation")
utils.print_dict(dict_valid)"""

model = Sequential()
model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.2,
               input_shape=(time_step, num_features), return_sequences=True))
model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.2))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse",
              metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])

model.fit(X_t, Y_t, epochs=10)

X_v = X_v.reshape((len(X_v), time_step, num_features))

y_test = model.predict(X_v)
print(y_test)
score = model.evaluate(X_v, Y_v, batch_size=16)
print(score)
