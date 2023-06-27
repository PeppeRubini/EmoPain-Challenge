import tkinter as tk
import cv2
from PIL import Image, ImageTk
from feat import Detector
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow import keras

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

            button = tk.Button(self.menu_frame,
                               image=clicked_button,
                               border=0,
                               cursor='hand2',
                               command=cmd,
                               relief=tk.SUNKEN,
                               bg="#F0FAFF")

            button.bind("<Enter>", on_enter)
            button.bind("<Leave>", on_leave)
            button.place(x=x, y=y)

        self.video_frame = tk.Frame(self.root, width=640, height=480, relief=tk.RIDGE, borderwidth=5, bg="#1F77B4")
        self.video_frame.place(x=10, y=10)

        self.video_canvas = tk.Canvas(self.video_frame, width=640, height=480, bg="#E6EEF2")
        self.video_canvas.pack()

        self.au_frame = tk.Frame(self.root, width=450, height=350, relief=tk.RIDGE, borderwidth=5, bg="#1F77B4")
        self.au_frame.place(x=674, y=10)

        self.plot_au_canvas = tk.Canvas(self.au_frame, width=450, height=350, bg="#E6EEF2")
        self.plot_au_canvas.pack()

        self.plot_frame = tk.Frame(self.root, width=450, height=350, relief=tk.RIDGE, borderwidth=5, bg="#1F77B4")
        self.plot_frame.place(x=674, y=384)

        self.plot_pain_canvas = tk.Canvas(self.plot_frame, width=450, height=350, bg="#E6EEF2")
        self.plot_pain_canvas.pack()

        self.menu_frame = tk.Frame(self.root, width=674, height=254, relief=tk.RIDGE, bg="#F0FAFF")
        self.menu_frame.place(x=0, y=504)

        create_button(115, 35, 'webcam button.png', 'webcam button clicked.png', self.open_webcam)
        create_button(410, 35, 'video button.png', 'video button clicked.png', self.open_video)

        self.fps_label = tk.Label(self.menu_frame)
        self.fps_label.config(font=('Helvetica', 18), fg="#00356A", bg="#F0FAFF")
        self.fps_label.place(x=124, y=150)

        self.pain_label = tk.Label(self.menu_frame)
        self.pain_label.config(font=('Helvetica', 18), fg="#00356A", bg="#F0FAFF")
        self.pain_label.place(x=376, y=150)

    def open_webcam(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.cap = cv2.VideoCapture(0)
        self.show_frame(self.start_time)

    def open_video(self):
        self.video_path = tk.filedialog.askopenfilename(initialdir="/", title="Select a Video",
                                                        filetypes=[("Video files", ["*.mp4", "*.mov", "*.wmv", "*.flv",
                                                                                    "*.avi", "*.mkv", "*.webm",
                                                                                    "*.m4v"])])
        self.frame_count = 0
        self.start_time = time.time()
        self.cap = cv2.VideoCapture(self.video_path)
        self.show_frame(self.start_time)

    def show_frame(self, st):
        # start_time = time.time()
        ret, frame = self.cap.read()
        try:
            if frame.shape[1] / frame.shape[0] != 4 / 3:
                frame = make_4_3(frame)
                frame = cv2.resize(frame, (640, 480))
        except:
            print("Frame finished")
        self.frame_count += 1
        plotting = True
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.image = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(self.image)
            if self.frame_count % 10 == 0:
                try:
                    faces = detector.detect_faces(frame, threshold=0.5)
                    landmarks = detector.detect_landmarks(frame, faces)
                    aus = detector.detect_aus(frame, landmarks)
                    df_au = pd.DataFrame(columns=au_r_list)
                    au_row = np.delete(aus[0][0] * 5, indices)
                    df_au.loc[len(df_au)] = au_row
                    # todo creare ciclo per prendere un numero di frame pari a timestep (eliminare le due righe sottostanti)
                    df_au.loc[len(df_au)] = au_row
                    df_au.loc[len(df_au)] = au_row
                    a = df_au.to_numpy()
                    a = a.reshape(1, 3, 17)
                    pain = model.predict(a)
                    print(pain[0][0])
                except:
                    print('Error in pipeline of pain intensity extraction')
                    plotting = False
            else:
                time.sleep(0.025)
                plotting = False

            if plotting:
                self.pain_label.config(text=f"PAIN LEVEL: {pain[0][0]:.2f}")
                self.pain_list.append(pain[0][0])
                self.frame_count_list.append(self.frame_count)

                plt.figure()
                plt.plot(self.frame_count_list, self.pain_list)
                plt.ylim([0, 3])
                plt.xlabel('frame')
                plt.ylabel('pain')
                plt.grid(True)
                plt.savefig('./graphs/pain_graph.png')
                plt.close()

                self.image_g = Image.open('./graphs/pain_graph.png')
                self.photo_g = ImageTk.PhotoImage(self.image_g.resize((450, 350)))
                self.plot_pain_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_g)
                self.plot_pain_canvas.image = self.photo_g

                plt.bar(au_list, au_row)
                plt.title('Action Units')
                plt.savefig('./graphs/au_bar_graph.png')
                plt.close()

                self.image_au = Image.open('./graphs/au_bar_graph.png')
                self.photo_au = ImageTk.PhotoImage(self.image_au.resize((450, 350)))
                self.plot_au_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_au)
                self.plot_au_canvas.image = self.photo_au

            elapsed_time = time.time() - st
            fps = self.frame_count / elapsed_time
            self.fps_label.config(text=f"FPS: {fps:.2f}")

            # Update the canvas with the new frame
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # Schedule the next frame update
            self.video_canvas.after(15, self.show_frame, self.start_time)


model = keras.models.load_model('pain_model/modello.h5')
detector = Detector(face_model=face_model, landmark_model=landmark_model, au_model=au_model)
root = tk.Tk()
app = App(root)
root.mainloop()
