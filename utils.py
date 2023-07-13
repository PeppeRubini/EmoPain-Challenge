import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np


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
