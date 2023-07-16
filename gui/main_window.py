import tkinter as tk
from lstm_gui import lstm_gui
from svr_gui import svr_gui
from home_gui import home_gui

pages = {
    "home_gui": home_gui,
    "lstm_gui": lstm_gui,
    "svr_gui": svr_gui
}


class main_window(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame = None
        self.switch_frame("home_gui")
        self.lift()

    def switch_frame(self, page_name):
        """Destroys current frame and replaces it with a new one."""
        cls = pages[page_name]
        new_frame = cls(root=self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()
        return self._frame


if __name__ == "__main__":
    app = main_window()
    app.mainloop()
