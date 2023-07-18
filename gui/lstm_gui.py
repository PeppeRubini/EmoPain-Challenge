import os
import threading
import tkinter as tk
from io import BytesIO
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk
from feat import Detector
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow import keras
import warnings
from matplotlib import use
from collections import deque
from utils import center_window, create_button, gauge_chart, make_4_3

use('agg')
warnings.filterwarnings("ignore", category=UserWarning)
font = ImageFont.truetype("./font/Helvetica.ttf", 16)

au_r_list = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
             'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
             'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
au_list = ['01', '02', '04', '05', '06', '07', '09', '10', '12', '14', '15', '17', '20', '23', '25', '26', '45']
face_model = "faceboxes"
landmark_model = "mobilefacenet"
au_model = "xgb"
indices = [8, 15, 18]
time_step = 90
overlapping = 0.5


def kill():
    if tk.messagebox.askyesno("Pain Detector", "Are you sure you want to exit?"):
        os._exit(1)


class lstm_gui(tk.Frame):
    def __init__(self, root):
        tk.Frame.__init__(self, root, width=1132, height=785, bg='#F0FAFF')

        root.protocol("WM_DELETE_WINDOW", kill)
        self.pain_list = []
        self.frame_count_list = []
        self.frame_count = 0
        self.video_path = ''
        self.image_path = ''
        self.image = None
        self.photo = None
        self.image_g = None
        self.photo_g = None
        self.image_au = None
        self.photo_au = None
        self.cap = None
        self.start_time = None
        self.root = root
        self.root.title("Pain Detector")
        center_window(self.root, 1132, 785)
        self.root.configure(bg='#F0FAFF')
        self.root.resizable(False, False)

        self.frame_switched = False
        self.current_frame = "lstm_gui"

        self.menu_frame = tk.Frame(self.root, width=1132, height=50, relief=tk.RIDGE, bg='#F0FAFF')
        self.menu_frame.place(x=0, y=0)

        create_button(5, 5, self.menu_frame, 'webcam button.png', 'webcam button clicked.png', self.open_webcam)
        create_button(135, 5, self.menu_frame, 'video button.png', 'video button clicked.png', self.open_video)
        create_button(1000, 5, self.menu_frame, 'change to svr button.png', 'change to svr button clicked.png',
                      lambda: [self.set_frame_switched(True), root.switch_frame("svr_gui")])

        self.video_frame = tk.Frame(self.root, width=640, height=480, relief=tk.RIDGE, borderwidth=3, bg='#1F77B4')
        self.video_frame.place(x=7, y=50)

        self.video_canvas = tk.Canvas(self.video_frame, width=640, height=480, bg='#E6EEF2')
        self.video_canvas.pack()

        self.video_label = tk.Label(self.video_canvas, text="No video source", font=('Helvetica', 16, 'bold'),
                                    fg='#1F77B4', bg='#E6EEF2')
        self.video_label.place(x=240, y=225)

        self.au_frame = tk.Frame(self.root, width=450, height=350, relief=tk.RIDGE, borderwidth=3, bg='#1F77B4')
        self.au_frame.place(x=664, y=50)

        self.photo_au = Image.open("./charts/au_empty.png")
        self.photo_au = ImageTk.PhotoImage(self.photo_au.resize((450, 350)))

        self.plot_au_canvas = tk.Canvas(self.au_frame, width=450, height=350, bg='#E6EEF2')
        self.plot_au_canvas.pack()

        self.plot_au_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_au)
        self.plot_au_canvas.image = self.photo_au

        self.plot_frame = tk.Frame(self.root, width=450, height=350, relief=tk.RIDGE, borderwidth=3, bg='#1F77B4')
        self.plot_frame.place(x=664, y=418)

        self.photo_pain = Image.open("./charts/trend_clean.png")
        self.photo_pain = ImageTk.PhotoImage(self.photo_pain.resize((450, 350)))

        self.plot_pain_canvas = tk.Canvas(self.plot_frame, width=450, height=350, bg='#E6EEF2')
        self.plot_pain_canvas.pack()

        self.plot_pain_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_pain)
        self.plot_pain_canvas.image = self.photo_pain

        self.plot_pain_label = tk.Label(self.plot_pain_canvas, text="No pain trend available for images",
                                        font=('Helvetica', 16, 'bold'), fg='#1F77B4', bg='#E6EEF2')

        self.gauge_frame = tk.Frame(self.root, width=640, height=220, relief=tk.RIDGE, borderwidth=3, bg='#1F77B4')
        self.gauge_frame.place(x=7, y=548)

        self.photo_gauge = Image.open("./charts/gauge_clean.png")
        self.photo_gauge = ImageTk.PhotoImage(self.photo_gauge)

        self.gauge_canvas = tk.Canvas(self.gauge_frame, width=640, height=220, bg='#E6EEF2')
        self.gauge_canvas.pack()

        self.gauge_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_gauge)
        self.gauge_canvas.image = self.photo_gauge

        # aggiunte per thread
        self.frame = None
        self.frame_queue = deque()
        self.plotting_au = False
        self.pain = None
        self.au_row = None
        self.au_row_queue = deque()
        self.df_au = pd.DataFrame(columns=au_r_list)
        self.extracting_feature = False
        self.n_prediction = 0
        self.thread_number = 0
        self.processing_image = False

    def set_frame_switched(self, boolean):
        self.frame_switched = boolean

    def reinitialize(self):
        self.plot_pain_label.place_forget()

        self.plot_au_canvas.image = self.photo_au

        self.plot_pain_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_pain)
        self.plot_pain_canvas.image = self.photo_pain

        self.gauge_canvas.image = self.photo_gauge

        self.pain_list = []
        self.frame_count_list = []

        self.frame_queue = deque()
        self.au_row_queue = deque()

        self.df_au = pd.DataFrame(columns=au_r_list)
        self.n_prediction = 0

    def open_webcam(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            self.set_frame_switched(True)
            new_frame = self.root.switch_frame(self.current_frame)
            new_frame.frame_count = 0
            new_frame.start_time = time.time()
            new_frame.cap = cv2.VideoCapture(0)
            new_frame.reinitialize()
            new_frame.show_frame(new_frame.start_time)
        else:
            tk.messagebox.showerror("Error", "No webcam found")

    def open_video(self):
        self.video_path = tk.filedialog.askopenfilename(initialdir="/", title="Select a Video",
                                                        filetypes=[("Video files", ["*.mp4", "*.mov", "*.wmv", "*.flv",
                                                                                    "*.avi", "*.mkv", "*.webm",
                                                                                    "*.m4v"])])
        if self.video_path != "":
            self.set_frame_switched(True)
            new_frame = self.root.switch_frame(self.current_frame)
            new_frame.frame_count = 0
            new_frame.start_time = time.time()
            new_frame.cap = cv2.VideoCapture(self.video_path)
            new_frame.reinitialize()
            new_frame.show_frame(new_frame.start_time)

    def plot_au(self):
        background = Image.open("./charts/au_empty.png")
        bar = Image.open("./charts/au_bar.png")
        x_begin = 103
        y_begin = 59
        space = 6
        width = bar.size[0]
        au_row = self.au_row_queue.popleft()
        i = 0
        while i < len(au_row):
            y_resize = int(368 * au_row[i] / 5)
            if y_resize != 0:
                bar_resized = bar.resize((width, y_resize))
                background.paste(bar_resized, (x_begin + (space + width) * i, y_begin + (bar.size[1] - y_resize)),
                                 mask=bar_resized)
            i += 1

        self.plotting_au = False
        self.image_au = background
        self.photo_au = ImageTk.PhotoImage(self.image_au.resize((450, 350)))
        self.plot_au_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_au)
        self.plot_au_canvas.image = self.photo_au

    def feature_extraction(self):
        try:
            self.extracting_feature = True
            frame = self.frame_queue.popleft()
            faces = detector.detect_faces(frame, threshold=0.5)
            landmarks = detector.detect_landmarks(frame, faces)
            aus = detector.detect_aus(frame, landmarks)
            self.au_row = np.delete(aus[0][0] * 5, indices)
            self.au_row_queue.append(self.au_row)
            self.thread_number -= 1
            if len(self.au_row_queue) > 0 and not self.plotting_au:
                self.plotting_au = True
                threading.Thread(target=self.plot_au).start()
            self.df_au = pd.concat([self.df_au, pd.DataFrame(self.au_row, index=au_r_list).transpose()], axis=0,
                                   ignore_index=True)
            # print(f"****feature estratte {self.df_au.shape[0]}")
            if self.df_au.shape[0] == time_step:
                threading.Thread(target=self.predict).start()
            self.extracting_feature = False
        except:
            print('Error in pipeline of pain intensity extraction')
            self.extracting_feature = False

    def predict(self):
        # controlla se la lista è vuota
        if not self.pain_list:
            current_value = 0
        else:
            current_value = self.pain_list[-1]

        a = self.df_au[:90].to_numpy()
        a = a.reshape(1, time_step, 17)
        pain = model.predict(a)
        self.n_prediction += 1
        # print(pain[0][0])

        prediction = round(pain[0][0], 2)

        self.pain_list.append(pain[0][0])
        self.frame_count_list.append(self.n_prediction)

        # grafico pain label

        plt.figure()
        plt.plot(self.frame_count_list, self.pain_list, marker='o', linewidth=2)
        plt.xlim(left=1)
        plt.xscale('linear')
        plt.ylim([0, 3])
        plt.xlabel('prediction')
        plt.ylabel('pain')
        plt.title('Pain Trend')
        plt.grid(True)
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        self.plot_pain_label.place_forget()
        self.image_g = Image.open(buffer)
        self.photo_g = ImageTk.PhotoImage(self.image_g.resize((450, 350)))
        self.plot_pain_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_g)
        self.plot_pain_canvas.image = self.photo_g

        # grafico pain label (gauge)
        n = 10
        step = abs(current_value - prediction) / n
        if current_value > prediction:
            step *= -1
        i = 0
        for x in np.arange(current_value, prediction, step):
            i += 1
            if i == n:
                x = prediction

            image_gauge = gauge_chart(x)

            photo_gauge = ImageTk.PhotoImage(image_gauge)
            self.gauge_canvas.create_image(0, 0, anchor=tk.NW, image=photo_gauge)
            self.gauge_canvas.image = photo_gauge

        self.df_au.drop(index=self.df_au.index[:int(time_step * overlapping)], axis=0, inplace=True)

    def video_ended(self):

        while len(self.frame_queue) > time_step:
            if self.thread_number <= 12:
                threading.Thread(target=self.feature_extraction).start()
                self.thread_number += 1
            time.sleep(0.1)
            if self.plotting_au:
                time.sleep(0.1)

    def show_frame(self, st):
        # start_time = time.time()
        ret, self.frame = self.cap.read()
        try:
            if self.frame.shape[1] / self.frame.shape[0] != 4 / 3:
                self.frame = make_4_3(self.frame)
                self.frame = cv2.resize(self.frame, (640, 480))
            self.video_label.place_forget()
        except:
            print("Frame finished")
            threading.Thread(target=self.video_ended).start()
        self.frame_count += 1
        if ret:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.image = Image.fromarray(self.frame)
            elapsed_time = time.time() - st
            fps = self.frame_count / elapsed_time
            ImageDraw.Draw(self.image).text((555, 5), f"FPS: {fps:.2f}", fill='#1F77B4', font=font)
            self.photo = ImageTk.PhotoImage(self.image)

            self.frame_queue.append(self.frame)
            # print(f"frame estratti {self.frame_count}")
            # Update the canvas with the new frame
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            if self.thread_number <= 12:
                threading.Thread(target=self.feature_extraction).start()
                self.thread_number += 1
            # Schedule the next frame update
            # after_idle sembra essere più veloce di after
            # self.video_canvas.after(1, self.show_frame, self.start_time)
            if not self.frame_switched:
                self.video_canvas.after_idle(self.show_frame, self.start_time)


model = keras.models.load_model('../pain_model/lstm.h5')
detector = Detector(face_model=face_model, landmark_model=landmark_model, au_model=au_model)
