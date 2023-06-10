import pandas as pd
import numpy as np
import os

label_data_path_train = "./Dataset/Pain Labels/train/"
geometric_data_path_train = "./Dataset/Face Features/Geometric Features/train/"
#label_data_path_valid = "./Dataset/Pain Labels/valid/"
#geometric_data_path_valid = "./Dataset/Face Features/Geometric Features/valid/"
for file in os.listdir(label_data_path_train):
    csv_label = pd.read_csv(label_data_path_train + file)

    for label_value in csv_label:
        print(label_value)
        if int(label_value) > 0:
            #crea file
            print("c'è dolore")
    #print(csv_label)
    #csv_geo = pd.read_csv(geometric_data_path_train + file)
    #print(csv_geo)
