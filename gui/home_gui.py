import tkinter as tk
from utils import center_window, create_button


class home_gui(tk.Frame):
    def __init__(self, root):
        tk.Frame.__init__(self, root, width=450, height=200, bg='#F0FAFF')

        self.root = root

        self.root.title("Pain Detector")
        center_window(self.root, 450, 200)
        self.root.configure(bg='#F0FAFF')
        self.root.resizable(False, False)

        self.label = tk.Label(self, text="Select the model for the prediction", font="Helvetica 16 bold", fg='#1F77B4',
                              bg='#F0FAFF')
        self.label.place(x=46, y=35)

        create_button(60, 100, self, 'lstm button.png', 'lstm button clicked.png',
                      lambda: root.switch_frame("lstm_gui"))
        create_button(253, 100, self, 'svr button.png', 'svr button clicked.png',
                      lambda: root.switch_frame("svr_gui"))