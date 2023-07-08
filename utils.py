import tkinter as tk
from PIL import Image, ImageTk


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
