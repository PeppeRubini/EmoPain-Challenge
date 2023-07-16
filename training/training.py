from utils import make_step_dataset, label_dict, pie_chart_label, print_dict
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.regularizers import l2
from correlation_coefficient import *

path_clean_dataset_train = "./CleanDataset/train/"
path_clean_dataset_valid = "./CleanDataset/valid/"

time_step = 90
num_features = 17
overlapping_rate = 0.5

x_train, y_train = make_step_dataset(path_clean_dataset_train, time_step, overlapping_rate)
x_valid, y_valid = make_step_dataset(path_clean_dataset_valid, time_step, overlapping_rate)

if input("Do you want to see the graphs? (y/n)") == "y":
    dict_train = label_dict(y_train)
    dict_valid = label_dict(y_valid)
    print("\ntraining")
    print_dict(dict_train)
    print("\nvalidation")
    print_dict(dict_valid)
    pie_chart_label(label_dict(y_train))
    pie_chart_label(label_dict(y_valid))

model = Sequential()
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, input_shape=(time_step, num_features), return_sequences=True,
               kernel_regularizer=l2(0.01)))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.01)))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="relu"))

model.compile(optimizer="adam", loss="mse",
              metrics=['accuracy', tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
model.fit(x_train, y_train, epochs=15, verbose=1)
x_valid = x_valid.reshape((len(x_valid), time_step, num_features))
score = model.evaluate(x_valid, y_valid)

y_pred = model.predict(x_valid, verbose=0)
predicted = []
real = []
for i in range(y_pred.shape[0]):
    predicted.append(y_pred[i][0])
    real.append(np.float32(y_valid[i][0]))
    # print(f"{y_pred[i][0]} => {y_valid[i][0]}")

print("ACCURACY:\t", score[1])
print("RMSE:\t\t", score[2])
print("MAE:\t\t", score[3])
print("PCC\t\t\t", pcc(tf.convert_to_tensor(real), tf.convert_to_tensor(predicted)))
print("CCC\t\t\t", ccc(tf.convert_to_tensor(real), tf.convert_to_tensor(predicted)))

if input("Do you want to save the model? (y/n)") == "y":
    name = input("Insert the name of the model: ")
    model.save("../pain_model/" + name + ".h5")
    print("Model saved")
