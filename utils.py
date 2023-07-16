import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont


def label_dict_bin(array):
    d = {"0": 0, "1": 0}
    for y in array:
        if y == 0:
            d["0"] += 1
        else:
            d["1"] += 1
    return d


def label_dict(array):
    d = {"0": 0, "1": 0, "2": 0, "3": 0}
    for y in array:
        if y == 0:
            d["0"] += 1
        elif y == 1:
            d["1"] += 1
        elif y == 2:
            d["2"] += 1
        elif y == 3:
            d["3"] += 1
    return d


def make_step_dataset(path_dataset, step, overlapping_rate):
    X = []
    Y = []
    for file in os.listdir(path_dataset):
        df_label = pd.DataFrame()
        df_geo = pd.DataFrame()
        table = pd.read_csv(path_dataset + file)
        df_label = pd.concat([df_label, table.pop("pain_label")])
        df_geo = pd.concat([df_geo, table])
        x = df_geo.to_numpy()
        y = df_label.to_numpy().astype(np.int64)
        d = label_dict_bin(y)
        tot = d.get("1") + d.get("0")
        zero_rate = d.get('0') / tot
        if zero_rate <= 0.50:
            overlapping = int(step * overlapping_rate)
            start = 0
            end = start + step
            while end < len(x):
                l = x[start:end]
                start = end - overlapping
                end = start + step
                X.append(l)
                Y.append(y[end - overlapping])
    return np.array(X), np.array(Y)


def print_dict(d):
    for k, v in d.items():
        print(k, v)
        # print(v)


def pie_chart_label(dictionary):
    labels = []
    sizes = []
    for x, y in dictionary.items():
        labels.append(x)
        sizes.append(y)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()


def center_window(window, app_width, app_height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight() - 80
    x = (screen_width / 2) - (app_width / 2)
    y = (screen_height / 2) - (app_height / 2)
    window.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')


def create_button(x, y, frame, img1, img2, cmd):
    base_button = ImageTk.PhotoImage(Image.open('./buttons/' + img1))
    clicked_button = ImageTk.PhotoImage(Image.open('./buttons/' + img2))

    def on_enter(event):
        button['image'] = clicked_button

    def on_leave(event):
        button['image'] = base_button

    button = tk.Button(frame, image=clicked_button, border=0, cursor='hand2', command=cmd,
                       relief=tk.SUNKEN, bg="#F0FAFF")
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)
    button.place(x=x, y=y)


def gauge_chart(prediction):
    percent = prediction / 3
    rotation = 180 * percent
    rotation = 90 - rotation

    gauge = Image.open('./charts/grafico.png')
    x = gauge.size[0]
    y = gauge.size[1]
    const = 70
    loc = (int(x / 2), y - const)

    needle = Image.open('./charts/indicatore.png')
    needle = needle.rotate(rotation, resample=Image.BICUBIC, center=loc)
    gauge.paste(needle, mask=needle)

    font = ImageFont.truetype("./font/Helvetica.ttf", 30)
    ImageDraw.Draw(gauge).text((int(x / 2) - 30, int(y / 2) + 180), f"{prediction:.2f}", fill='white', font=font)
    gauge.thumbnail((640, 220), Image.ANTIALIAS)

    layer = Image.new("RGB", (640, 220), (255, 255, 255))
    layer.paste(gauge, tuple(map(lambda x: int((x[0] - x[1]) / 2), zip((640, 220), gauge.size))))

    return layer


def make_4_3(frame):
    if frame.shape[0] > frame.shape[1]:
        x = abs(frame.shape[1] - (int(frame.shape[1] * 3 / 4)))
        new_frame = np.pad(frame, ((0, 0), (int(x * 2), int(x * 2)), (0, 0)), 'constant')
    else:
        y = abs((int((frame.shape[0] * 4 / 3) - frame.shape[0])))
        new_frame = np.pad(frame, ((int(y / 2), int(y / 2)), (0, 0), (0, 0)), 'constant')
    return new_frame
