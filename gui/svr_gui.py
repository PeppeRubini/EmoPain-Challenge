import threading
import time
import tkinter as tk
from io import BytesIO
import cv2
from PIL import Image, ImageFont, ImageTk
from feat import Detector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import use
from joblib import load
from utils import create_button, gauge_chart, make_4_3
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
                      lambda: [self.set_frame_switched(True), root.switch_frame("lstm_gui")])

        self.trend = False
        self.current_frame = "svr_gui"

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
            current_value = self.pain_list[-1]
        if len(self.df_au) == 0:
            return

        pain = model.predict(self.df_au.head(1))
        self.n_prediction += 1
        # print(pain[0])
        try:
            self.df_au.drop(self.df_au.index[0], inplace=True)
        except:
            pass

        prediction = round(pain[0], 2)

        self.pain_list.append(pain[0])
        self.frame_count_list.append(self.n_prediction)
        # grafico pain label
        if not self.trend:
            if not self.processing_image:
                self.trend = True
                plt.figure()
                if len(self.pain_list) < 30:
                    plt.plot(self.frame_count_list, self.pain_list, marker='o', linewidth=2)
                else:
                    plt.plot(self.frame_count_list, self.pain_list, linewidth=2)
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
                self.trend = False
                self.image_g = Image.open(buffer)
                self.photo_g = ImageTk.PhotoImage(self.image_g.resize((450, 350)))
                self.plot_pain_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_g)
                self.plot_pain_canvas.image = self.photo_g
            else:
                self.processing_image = False

        # grafico pain label (gauge)
        n = 2
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

    def video_ended(self):
        while len(self.frame_queue) > 0:
            if self.thread_number <= 12:
                threading.Thread(target=self.feature_extraction).start()
                self.thread_number += 1
            time.sleep(0.1)
            if self.plotting_au:
                time.sleep(0.1)

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
            self.set_frame_switched(True)
            new_frame = self.root.switch_frame("svr_gui")
            new_frame.processing_image = True
            new_frame.reinitialize()
            new_frame.plot_pain_canvas.delete("all")
            new_frame.plot_pain_label.place(x=54, y=160)
            new_frame.frame = cv2.imread(self.image_path)
            if new_frame.frame.shape[1] / new_frame.frame.shape[0] != 4 / 3:
                new_frame.frame = make_4_3(new_frame.frame)
                new_frame.frame = cv2.resize(new_frame.frame, (640, 480))
            new_frame.video_label.place_forget()
            new_frame.frame = cv2.cvtColor(new_frame.frame, cv2.COLOR_BGR2RGB)
            new_frame.image = Image.fromarray(new_frame.frame)
            new_frame.photo = ImageTk.PhotoImage(new_frame.image)
            new_frame.frame_queue.append(new_frame.frame)
            new_frame.video_canvas.create_image(0, 0, anchor=tk.NW, image=new_frame.photo)
            if not new_frame.extracting_feature:
                threading.Thread(target=new_frame.feature_extraction).start()


model = load('../pain_model/svr.pkl')
detector = Detector(face_model=face_model, landmark_model=landmark_model, au_model=au_model)
