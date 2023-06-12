import pandas as pd
import os

header_list = ['frame', ' face_id', ' timestamp', ' confidence', ' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r',
               ' AU07_r',
               ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r',
               ' AU26_r',
               ' AU45_r', ' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', ' AU10_c',
               ' AU12_c',
               ' AU14_c', ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c', ' AU28_c', ' AU45_c']

label_data_path_train = "./Dataset/Pain Labels/train/"
geometric_data_path_train = "./Dataset/Face Features/Geometric Features/train/"
label_data_path_valid = "./Dataset/Pain Labels/valid/"
geometric_data_path_valid = "./Dataset/Face Features/Geometric Features/valid/"

def clean(label_path, geometric_path):
    for file in os.listdir(label_path):
        # metto insieme le due tabelle
        df_geo = pd.read_csv(geometric_path + file)
        df_label = pd.read_csv(label_path + file, header=None)
        df_label.columns = ['label']
        df_merged = pd.concat([df_geo, df_label], axis=1)
        # filtro le righe inutili
        df_merged_clean = df_merged[(df_merged[' success'] != 0) & (df_merged['label'] > 0)]
        if not df_merged_clean.empty:
            # salvo il file delle pain lable
            df_label_clean = df_merged_clean.pop('label')
            df_label_clean.to_csv(str(label_path).replace("Dataset", "CleanDataset") + file, index=False, header=None)
            # salvo il file delle feature
            df_geo_clean = df_merged_clean[header_list]
            df_geo_clean.to_csv(str(geometric_path).replace("Dataset", "CleanDataset") + file, index=False)

clean(label_data_path_train, geometric_data_path_train)
clean(label_data_path_valid, geometric_data_path_valid)
