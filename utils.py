import tkinter as tk
from PIL import Image, ImageTk


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
