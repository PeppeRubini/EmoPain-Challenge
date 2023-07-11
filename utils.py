import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import math
from io import BytesIO


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
    LOW = "#8fbbd9"
    MEDIUM = "#629fca"
    HIGH = "#1f77b4"

    colors = [LOW, MEDIUM, HIGH]
    values = [0, 1, 2, 3]

    fig = plt.figure(figsize=(5, 5))  # immagine a 18x18 inches
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)

    ax.bar(x=[0, math.pi / 3, 2 * (math.pi / 3)], width=1.05, height=0.5, bottom=2,
           linewidth=3, edgecolor="white",
           color=colors, align="edge")

    # label per ogni fascia
    for loc, val in zip([0, math.pi / 3, 2 * (math.pi / 3) - 0.02, math.pi], values):
        ax.annotate(val, xy=(loc, 2.5), ha="right" if val < 2 else "left")

    # indicatore
    plt.annotate(f'{prediction:.2f}', xytext=(0, 0), xy=((3.14 / 3) * prediction, 2.0),
                 arrowprops=dict(arrowstyle="wedge, tail_width=0.5", color="#144c73", shrinkA=0),
                 bbox=dict(boxstyle="circle", facecolor="#144c73", linewidth=0),
                 fontsize=10, color="white", ha="center"
                 )

    ax.set_axis_off()

    buffer_gauge = BytesIO()
    plt.savefig(buffer_gauge, format='png')
    plt.close()
    return buffer_gauge


def make_4_3(frame):
    if frame.shape[0] > frame.shape[1]:
        x = abs(frame.shape[1] - (int(frame.shape[1] * 3 / 4)))
        new_frame = np.pad(frame, ((0, 0), (int(x * 2), int(x * 2)), (0, 0)), 'constant')
    else:
        y = abs((int((frame.shape[0] * 4 / 3) - frame.shape[0])))
        new_frame = np.pad(frame, ((int(y / 2), int(y / 2)), (0, 0), (0, 0)), 'constant')
    return new_frame
