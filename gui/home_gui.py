import tkinter as tk
from PIL import ImageTk, Image
from lstm_gui import lstm_gui
from utils import center_window, create_button


class home_gui:
    def __init__(self, root):
        self.root = root
        self.root.title("Pain Detector")
        center_window(self.root, 450, 200)
        self.root.configure(bg='#F0FAFF')
        self.root.resizable(False, False)

        self.label = tk.Label(root, text="Select the model for the prediction", font="Helvetica 16 bold", fg='#1F77B4',
                              bg='#F0FAFF')
        self.label.place(x=46, y=35)

        create_button(60, 100, self.root, 'lstm button.png', 'lstm button clicked.png',
                      self.launch_lstm_window)
        create_button(253, 100, self.root, 'svr button.png', 'svr button clicked.png',
                      self.launch_svr_window)

    def launch_lstm_window(self):
        self.root.withdraw()
        lstm_gui(tk.Toplevel(root))

    def launch_svr_window(self):
        self.root.withdraw()
        svr_gui(tk.Toplevel(root))


root = tk.Tk()
home_gui = home_gui(root)
root.mainloop()
