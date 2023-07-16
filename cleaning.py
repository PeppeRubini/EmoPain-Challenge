import pandas as pd
import os

header_list = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r',
               ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r', 'pain_label']

label_data_path_train = "./Dataset/Pain Labels/train/"
geometric_data_path_train = "./Dataset/Face Features/Geometric Features/train/"
label_data_path_valid = "./Dataset/Pain Labels/valid/"
geometric_data_path_valid = "./Dataset/Face Features/Geometric Features/valid/"

time_step = 90


def clean(label_path, geometric_path):
    for file in os.listdir(label_path):
        if file.startswith("P"):
            # merge table
            df_geo = pd.read_csv(geometric_path + file)
            df_label = pd.read_csv(label_path + file, header=None)
            df_label.columns = ['pain_label']
            df_merged = pd.concat([df_geo, df_label], axis=1)
            df_merged = df_merged[header_list]
            # change label values
            df_merged['pain_label'].replace([1, 2, 3], 1, inplace=True)
            df_merged['pain_label'].replace([4, 5, 6], 2, inplace=True)
            df_merged['pain_label'].replace([7, 8, 9, 10], 3, inplace=True)
            # saving
            df_merged.to_csv(str(label_path).replace("Dataset", "CleanDataset").replace("Pain Labels/", "") + file,
                             index=False)


print("cleaning training set")
clean(label_data_path_train, geometric_data_path_train)
print("training set cleaning done")
print("cleaning validation set")
clean(label_data_path_valid, geometric_data_path_valid)
print("validation set cleaning done")
