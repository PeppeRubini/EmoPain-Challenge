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
from utils import center_window, create_button
import math

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


class lstm_gui:
    def __init__(self, root):
        root.protocol("WM_DELETE_WINDOW", root.quit)
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

        self.menu_frame = tk.Frame(self.root, width=1132, height=50, relief=tk.RIDGE, bg='#F0FAFF')
        self.menu_frame.place(x=0, y=0)

        create_button(5, 5, self.menu_frame, 'webcam button.png', 'webcam button clicked.png', self.open_webcam)
        create_button(135, 5, self.menu_frame, 'video button.png', 'video button clicked.png', self.open_video)
        create_button(1000, 5, self.menu_frame, 'change to svr button.png', 'change to svr button clicked.png',
                      self.change_to_svr)

        self.video_frame = tk.Frame(self.root, width=640, height=480, relief=tk.RIDGE, borderwidth=3, bg='#1F77B4')
        self.video_frame.place(x=7, y=50)

        self.video_canvas = tk.Canvas(self.video_frame, width=640, height=480, bg='#E6EEF2')
        self.video_canvas.pack()

        self.video_label = tk.Label(self.video_canvas, text="No video source", font=('Helvetica', 16, 'bold'),
                                    fg='#1F77B4', bg='#E6EEF2')
        self.video_label.place(x=240, y=225)

        self.au_frame = tk.Frame(self.root, width=450, height=350, relief=tk.RIDGE, borderwidth=3, bg='#1F77B4')
        self.au_frame.place(x=664, y=50)

        self.plot_au_canvas = tk.Canvas(self.au_frame, width=450, height=350, bg='#E6EEF2')
        self.plot_au_canvas.pack()

        self.plot_au_label = tk.Label(self.plot_au_canvas, text="No action unit graph generated",
                                      font=('Helvetica', 16, 'bold'), fg='#1F77B4', bg='#E6EEF2')
        self.plot_au_label.place(x=70, y=160)

        self.plot_frame = tk.Frame(self.root, width=450, height=350, relief=tk.RIDGE, borderwidth=3, bg='#1F77B4')
        self.plot_frame.place(x=664, y=418)

        self.plot_pain_canvas = tk.Canvas(self.plot_frame, width=450, height=350, bg='#E6EEF2')
        self.plot_pain_canvas.pack()

        self.plot_pain_label = tk.Label(self.plot_pain_canvas, text="No pain graph generated",
                                        font=('Helvetica', 16, 'bold'), fg='#1F77B4', bg='#E6EEF2')
        self.plot_pain_label.place(x=103, y=160)

        self.gauge_frame = tk.Frame(self.root, width=640, height=220, relief=tk.RIDGE, borderwidth=3, bg='#1F77B4')
        self.gauge_frame.place(x=7, y=548)

        self.gauge_canvas = tk.Canvas(self.gauge_frame, width=640, height=220, bg='#E6EEF2')
        self.gauge_canvas.pack()

        self.gauge_label = tk.Label(self.gauge_canvas, text="No prediction generated",
                                    font=('Helvetica', 16, 'bold'), fg='#1F77B4', bg='#E6EEF2')
        self.gauge_label.place(x=200, y=95)

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
        self.animation = False

    def reinitialize(self):
        self.plot_au_canvas.delete('all')
        self.plot_pain_canvas.delete('all')
        self.gauge_canvas.delete('all')

        self.plot_au_label.place(x=70, y=160)
        self.plot_pain_label.place(x=103, y=160)
        self.gauge_label.place(x=200, y=95)

        self.pain_list = []
        self.frame_count_list = []

        self.frame_queue = deque()
        self.au_row_queue = deque()

        self.df_au = pd.DataFrame(columns=au_r_list)
        self.n_prediction = 0

    def open_webcam(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            self.frame_count = 0
            self.start_time = time.time()
            self.cap = cv2.VideoCapture(0)
            self.reinitialize()
            self.show_frame(self.start_time)
        else:
            tk.messagebox.showerror("Error", "No webcam found")

    def open_video(self):
        self.video_path = tk.filedialog.askopenfilename(initialdir="/", title="Select a Video",
                                                        filetypes=[("Video files", ["*.mp4", "*.mov", "*.wmv", "*.flv",
                                                                                    "*.avi", "*.mkv", "*.webm",
                                                                                    "*.m4v"])])
        if self.video_path != "":
            self.frame_count = 0
            self.start_time = time.time()
            self.cap = cv2.VideoCapture(self.video_path)
            self.reinitialize()
            self.show_frame(self.start_time)

    def change_to_svr(self):
        print("Hola")
        """self.root.destroy()
        self.root = tk.Tk()
        self.app = SVR(self.root)"""

    def plot_au(self):
        while self.animation:
            time.sleep(0.001)
            if not self.animation:
                break
        plt.bar(au_list, self.au_row_queue.popleft())
        plt.title('Action Units')
        plt.ylim(0, 5)
        plt.yscale('linear')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()

        self.plotting_au = False
        self.plot_au_label.place_forget()
        self.image_au = Image.open(buffer)
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
        print(self.pain_list)
        # controlla se la lista è vuota
        if not self.pain_list:
            current_value = 0
        else:
            current_value = self.pain_list[len(self.pain_list) - 1]

        a = self.df_au.to_numpy()
        a = a.reshape(1, time_step, 17)
        pain = model.predict(a)
        self.n_prediction += 1
        print(pain[0][0])

        prediction = round(pain[0][0], 2)

        self.pain_list.append(pain[0][0])
        self.frame_count_list.append(self.n_prediction)

        while self.plotting_au:
            time.sleep(0.001)
            if not self.plotting_au:
                break
        # grafico pain label
        plt.figure()
        plt.plot(self.frame_count_list, self.pain_list, marker='o')
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
        self.gauge_label.place_forget()
        i = 0
        for x in np.arange(current_value, prediction, step):
            i += 1
            if i == n:
                x = prediction
            while self.plotting_au:
                time.sleep(0.001)
                if not self.plotting_au:
                    break
            self.animation = True
            image_gauge = Image.open(gauge_chart(x))
            self.animation = False
            width, height = image_gauge.size
            image_gauge = image_gauge.crop(
                (0, 0, width, height / 2 + 75))  # crop: left, upper, right, lower corners of the box
            layer = Image.new("RGB", (640, 220), (255, 255, 255))
            layer.paste(image_gauge,
                        tuple(map(lambda x: int((x[0] - x[1]) / 2), zip((640, 220), image_gauge.size))))
            image_gauge = layer
            photo_gauge = ImageTk.PhotoImage(image_gauge)
            self.gauge_canvas.create_image(0, 0, anchor=tk.NW, image=photo_gauge)
            self.gauge_canvas.image = photo_gauge
        self.animation = False
        self.df_au.drop(index=self.df_au.index[:int(time_step * overlapping)], axis=0, inplace=True)

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
            if not self.extracting_feature:
                threading.Thread(target=self.feature_extraction).start()
            # Schedule the next frame update
            # after_idle sembra essere più veloce di after
            # self.video_canvas.after(1, self.show_frame, self.start_time)
            self.video_canvas.after_idle(self.show_frame, self.start_time)


model = keras.models.load_model('../pain_model/modello90-05_1.h5')
detector = Detector(face_model=face_model, landmark_model=landmark_model, au_model=au_model)
