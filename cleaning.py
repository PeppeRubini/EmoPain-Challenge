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
# label_data_path_valid = "./Dataset/Pain Labels/valid/"
# geometric_data_path_valid = "./Dataset/Face Features/Geometric Features/valid/"

# todo rendere questa una funzione a cui si passa il path delle pain lable e delle feature geometriche
#  in modo da riutilizzarla per train e validation set
for file in os.listdir(label_data_path_train):
    # metto insieme le due tabelle
    df_geo = pd.read_csv(geometric_data_path_train + file)
    df_label = pd.read_csv(label_data_path_train + file, header=None)
    df_label.columns = ['label']
    df_merged = pd.concat([df_geo, df_label], axis=1)
    # filtro le righe inutili
    df_merged_clean = df_merged[(df_merged[' success'] != 0) & (df_merged['label'] > 0)]
    # salvo il file delle pain lable todo cambiare percorso
    df_label_clean = df_merged_clean.pop('label')
    df_label_clean.to_csv(file, index=False, header=None)
    # salvo il file delle feature todo cambiare percorso
    df_geo_clean = df_merged_clean[header_list]
    df_geo_clean.to_csv(file, index=False)

"""    # elimino dati con success = 0
    df_geo_clean = df_geo[df_geo[' success'] != 0]
    # considero solo le colonne interessate
    df_geo_clean = df_geo_clean[header_list]
    # salvo todo creare le directory per il dataset pulito
    df_geo_clean.to_csv("ripulito.csv", index=False)"""
"""    df_label = pd.read_csv(label_data_path_train + file, header=None)  # header = None perchè non c'è intestazione
    i = 0
    for k, v in df_label.items():  # v è una series
        for val in v:
            if val > 0:
                print(val, file, i)  # creare file
            i += 1"""

"""
for label_value in df_label:
    print(label_value)
    if int(label_value) > 0:
        #crea file
        print("c'è dolore")
        """
# print(csv_label)
# csv_geo = pd.read_csv(geometric_data_path_train + file)
# print(csv_geo)
