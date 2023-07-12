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
import warnings
from matplotlib import use
from joblib import load
from collections import deque
from utils import center_window, create_button, gauge_chart, make_4_3
import math
from lstm_gui import lstm_gui

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


class svr_gui(lstm_gui):
    def __init__(self, root):
        super().__init__(root)
        create_button(265, 5, self.menu_frame, 'image button.png', 'image button clicked.png', self.open_image)
        create_button(1000, 5, self.menu_frame, 'change to lstm button.png', 'change to lstm button clicked.png',
                      self.change_to_lstm)

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
                # threading.Thread(target=self.plot_au).start()
                self.plot_au()
            self.df_au = pd.concat([self.df_au, pd.DataFrame(self.au_row, index=au_r_list).transpose()], axis=0,
                                   ignore_index=True)
            threading.Thread(target=self.predict).start()
            self.extracting_feature = False
        except:
            print("Error in pipeline of pain intensity extraction")
            self.extracting_feature = False

    def predict(self):
        # controlla se la lista è vuota
        if not self.pain_list:
            current_value = 0
        else:
            current_value = self.pain_list[len(self.pain_list) - 1]

        pain = model.predict(self.df_au.tail(1))
        self.n_prediction += 1
        print(pain[0])

        prediction = round(pain[0], 2)

        self.pain_list.append(pain[0])
        self.frame_count_list.append(self.n_prediction)

        """while self.plotting_au:
            time.sleep(0.001)
            if not self.plotting_au:
                break"""
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
        n = 2
        step = abs(current_value - prediction) / n
        if current_value > prediction:
            step *= -1
        self.gauge_label.place_forget()
        i = 0
        for x in np.arange(current_value, prediction, step):
            i += 1
            if i == n:
                x = prediction
            """while self.plotting_au:
                time.sleep(0.001)
                if not self.plotting_au:
                    break"""
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
        self.df_au.drop(self.df_au.index, inplace=True)

    def open_image(self):
        self.image_path = tk.filedialog.askopenfilename(initialdir="/", title="Select an Image",
                                                        filetypes=[("Image files", ["*.jpg", "*.png", "*.jpeg",
                                                                                    "*.bmp", "*.gif", "*.tiff",
                                                                                    "*.tif", "*.ppm", "*.pgm",
                                                                                    "*.pbm", "*.pnm", "*.svg",
                                                                                    "*.svgz", "*.ico", "*.eps",
                                                                                    "*.raw", "*.cr2", "*.nef",
                                                                                    "*.orf", "*.sr2", "*.raf",
                                                                                    "*.dng", "*.crw", "*.mef",
                                                                                    "*.webp"])])
        if self.image_path != "":
            self.reinitialize()
            self.frame = cv2.imread(self.image_path)
            if self.frame.shape[1] / self.frame.shape[0] != 4 / 3:
                self.frame = make_4_3(self.frame)
                self.frame = cv2.resize(self.frame, (640, 480))
            self.video_label.place_forget()
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.image = Image.fromarray(self.frame)
            self.photo = ImageTk.PhotoImage(self.image)
            self.frame_queue.append(self.frame)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            if not self.extracting_feature:
                threading.Thread(target=self.feature_extraction).start()

    def change_to_lstm(self):
        print("CIAO 2")
        """self.root.withdraw()
        lstm_gui(tk.Toplevel(root))"""


model = load('../pain_model/svr.pkl')
detector = Detector(face_model=face_model, landmark_model=landmark_model, au_model=au_model)