import threading
import tkinter as tk
from io import BytesIO
import cv2
from PIL import Image, ImageTk
from feat import Detector
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow import keras
import warnings
from matplotlib import use

use('agg')
warnings.filterwarnings("ignore", category=UserWarning)

au_r_list = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
             'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
             'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
au_list = ['01', '02', '04', '05', '06', '07', '09', '10', '12', '14', '15', '17', '20', '23', '25', '26', '45']
face_model = "retinaface"
landmark_model = "mobilenet"
au_model = "xgb"
indices = [8, 15, 18]


def make_4_3(frame):
    if frame.shape[0] > frame.shape[1]:
        x = abs(frame.shape[1] - (int(frame.shape[1] * 3 / 4)))
        new_frame = np.pad(frame, ((0, 0), (int(x * 2), int(x * 2)), (0, 0)), 'constant')
    else:
        y = abs((int((frame.shape[0] * 4 / 3) - frame.shape[0])))
        new_frame = np.pad(frame, ((int(y / 2), int(y / 2)), (0, 0), (0, 0)), 'constant')
    return new_frame


class App:
    def __init__(self, root):
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
        self.root.geometry("1148x758")
        self.root.configure(bg="#F0FAFF")
        self.root.resizable(False, False)

        def create_button(x, y, img1, img2, cmd):
            base_button = ImageTk.PhotoImage(Image.open('./buttons/' + img1))
            clicked_button = ImageTk.PhotoImage(Image.open('./buttons/' + img2))

            def on_enter(event):
                button['image'] = clicked_button

            def on_leave(event):
                button['image'] = base_button

            button = tk.Button(self.menu_frame, image=clicked_button, border=0, cursor='hand2', command=cmd,
                               relief=tk.SUNKEN, bg="#F0FAFF")
            button.bind("<Enter>", on_enter)
            button.bind("<Leave>", on_leave)
            button.place(x=x, y=y)

        self.video_frame = tk.Frame(self.root, width=640, height=480, relief=tk.RIDGE, borderwidth=5, bg="#1F77B4")
        self.video_frame.place(x=10, y=10)

        self.video_canvas = tk.Canvas(self.video_frame, width=640, height=480, bg="#E6EEF2")
        self.video_canvas.pack()

        self.video_label = tk.Label(self.video_canvas, text="No video source", font=('Helvetica', 16, 'bold'),
                                    fg="#1F77B4", bg="#E6EEF2")
        self.video_label.place(x=240, y=225)

        self.au_frame = tk.Frame(self.root, width=450, height=350, relief=tk.RIDGE, borderwidth=5, bg="#1F77B4")
        self.au_frame.place(x=674, y=10)

        self.plot_au_canvas = tk.Canvas(self.au_frame, width=450, height=350, bg="#E6EEF2")
        self.plot_au_canvas.pack()

        self.plot_au_label = tk.Label(self.plot_au_canvas, text="No action unit graph generated",
                                      font=('Helvetica', 16, 'bold'), fg="#1F77B4", bg="#E6EEF2")
        self.plot_au_label.place(x=70, y=160)

        self.plot_frame = tk.Frame(self.root, width=450, height=350, relief=tk.RIDGE, borderwidth=5, bg="#1F77B4")
        self.plot_frame.place(x=674, y=384)

        self.plot_pain_canvas = tk.Canvas(self.plot_frame, width=450, height=350, bg="#E6EEF2")
        self.plot_pain_canvas.pack()

        self.plot_pain_label = tk.Label(self.plot_pain_canvas, text="No pain graph generated",
                                        font=('Helvetica', 16, 'bold'), fg="#1F77B4", bg="#E6EEF2")
        self.plot_pain_label.place(x=103, y=160)

        self.menu_frame = tk.Frame(self.root, width=674, height=254, relief=tk.RIDGE, bg="#F0FAFF")
        self.menu_frame.place(x=0, y=504)

        create_button(115, 35, 'webcam button.png', 'webcam button clicked.png', self.open_webcam)
        create_button(410, 35, 'video button.png', 'video button clicked.png', self.open_video)

        self.fps_label = tk.Label(self.menu_frame, font=('Helvetica', 18), fg="#00356A", bg="#F0FAFF")
        self.fps_label.place(x=124, y=150)

        self.pain_label = tk.Label(self.menu_frame, font=('Helvetica', 18), fg="#00356A", bg="#F0FAFF")
        self.pain_label.place(x=376, y=150)

        # aggiunte per thread
        self.frame = None
        self.frame_list = []
        self.plotting_au = False
        self.pain = None
        self.au_row = None
        self.au_row_list = []
        self.df_au = pd.DataFrame(columns=au_r_list)
        self.extracting_feature = False
        self.n_prediction = 0

    def open_webcam(self):
        self.video_label.place_forget()
        self.frame_count = 0
        self.start_time = time.time()
        self.cap = cv2.VideoCapture(0)
        self.show_frame(self.start_time)

    def open_video(self):
        self.video_label.place_forget()
        self.video_path = tk.filedialog.askopenfilename(initialdir="/", title="Select a Video",
                                                        filetypes=[("Video files", ["*.mp4", "*.mov", "*.wmv", "*.flv",
                                                                                    "*.avi", "*.mkv", "*.webm",
                                                                                    "*.m4v"])])
        self.frame_count = 0
        self.start_time = time.time()
        self.cap = cv2.VideoCapture(self.video_path)
        self.show_frame(self.start_time)

    def plot_au(self):
        plt.bar(au_list, self.au_row_list.pop(0))
        plt.title('Action Units')
        plt.ylim(0, 5)
        plt.yscale('linear')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()

        self.plot_au_label.place_forget()
        self.image_au = Image.open(buffer)
        self.photo_au = ImageTk.PhotoImage(self.image_au.resize((450, 350)))
        self.plot_au_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_au)
        self.plot_au_canvas.image = self.photo_au
        self.plotting_au = False

    def feature_extraction(self):
        try:
            self.extracting_feature = True
            frame = self.frame_list.pop(0)
            faces = detector.detect_faces(frame, threshold=0.5)
            landmarks = detector.detect_landmarks(frame, faces)
            aus = detector.detect_aus(frame, landmarks)
            self.au_row = np.delete(aus[0][0] * 5, indices)
            self.au_row_list.append(self.au_row)
            if len(self.au_row_list) > 0 and not self.plotting_au:
                self.plotting_au = True
                threading.Thread(target=self.plot_au).start()
            self.df_au.loc[len(self.df_au)] = self.au_row
            print(f"****feature estratte {self.df_au.shape[0]}")
            if self.df_au.shape[0] == 150:
                threading.Thread(target=self.predict).start()
            self.extracting_feature = False
        except:
            print('Error in pipeline of pain intensity extraction')
            self.extracting_feature = False

    def predict(self):
        a = self.df_au.to_numpy()
        a = a.reshape(1, 150, 17)
        pain = model.predict(a)
        self.n_prediction += 1
        print(pain[0][0])
        self.pain_label.config(text=f"PAIN LEVEL: {float(pain[0][0]):.2f}")
        self.pain_list.append(pain[0][0])
        self.frame_count_list.append(self.n_prediction)

        plt.figure()
        plt.plot(self.frame_count_list, self.pain_list, marker='o')
        plt.xlim(left=1)
        plt.xscale('linear')
        plt.ylim([0, 3])
        plt.xlabel('prediction')
        plt.ylabel('pain')
        plt.grid(True)
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()

        self.plot_pain_label.place_forget()
        self.image_g = Image.open(buffer)
        self.photo_g = ImageTk.PhotoImage(self.image_g.resize((450, 350)))
        self.plot_pain_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_g)
        self.plot_pain_canvas.image = self.photo_g
        self.df_au = pd.DataFrame(columns=au_r_list)

    def show_frame(self, st):
        # start_time = time.time()
        ret, self.frame = self.cap.read()
        try:
            if self.frame.shape[1] / self.frame.shape[0] != 4 / 3:
                self.frame = make_4_3(self.frame)
                self.frame = cv2.resize(self.frame, (640, 480))
        except:
            print("Frame finished")
        self.frame_count += 1
        if ret:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.image = Image.fromarray(self.frame)
            self.photo = ImageTk.PhotoImage(self.image)

            self.frame_list.append(self.frame)
            # print(f"frame estratti {self.frame_count}")
            elapsed_time = time.time() - st
            fps = self.frame_count / elapsed_time
            self.fps_label.config(text=f"FPS: {fps:.2f}")
            # Update the canvas with the new frame
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            if not self.extracting_feature:
                threading.Thread(target=self.feature_extraction).start()
            # Schedule the next frame update
            # after_idle sembra essere più veloce di after
            # self.video_canvas.after(1, self.show_frame, self.start_time)
            self.video_canvas.after_idle(self.show_frame, self.start_time)


model = keras.models.load_model('pain_model/modello150.h5')
detector = Detector(face_model=face_model, landmark_model=landmark_model, au_model=au_model)
root = tk.Tk()
app = App(root)
root.mainloop()
