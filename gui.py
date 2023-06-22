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
        self.root.geometry("970x765")
        self.root.resizable(False, False)

        self.menu_frame = tk.Frame(self.root, width=1000, height=35, relief=tk.RAISED)
        self.menu_frame.place(x=0, y=0)

        self.video_frame = tk.Frame(self.root, width=500, height=510, relief=tk.RIDGE)
        self.video_frame.place(x=0, y=35)

        self.video_canvas = tk.Canvas(self.video_frame, width=500, height=510, bg='black')
        self.video_canvas.pack()

        self.pain_frame = tk.Frame(self.root, width=500, height=212, relief=tk.RIDGE)
        self.pain_frame.place(x=2, y=550)

        self.au_frame = tk.Frame(self.root, width=450, height=350, relief=tk.RIDGE, borderwidth=5)
        self.au_frame.place(x=505, y=35)

        self.plot_au_canvas = tk.Canvas(self.au_frame, width=450, height=350)
        self.plot_au_canvas.pack()

        self.plot_frame = tk.Frame(self.root, width=450, height=350, relief=tk.RIDGE, borderwidth=5)
        self.plot_frame.place(x=505, y=400)

        self.plot_pain_canvas = tk.Canvas(self.plot_frame, width=450, height=350)
        self.plot_pain_canvas.pack()

        self.webcam_button = tk.Button(self.menu_frame, text="Open Webcam", width=15, height=1, command=self.open_webcam)
        self.webcam_button.place(x=10, y=5)
        # todo aggiungere la funzionalità di apertura del file explorer
        self.file_button = tk.Button(self.menu_frame, text="Open File Explorer", width=15, height=1)
        self.file_button.place(x=130, y=5)
        self.video_button = tk.Button(self.menu_frame, text="Load Video", width=15, height=1, command=self.open_video)
        self.video_button.place(x=250, y=5)

        self.entry_video_path = tk.Entry(self.menu_frame, width=48)
        self.entry_video_path.insert(tk.END, "inserire/percorso/assoluto/file/video.mp4")
        self.entry_video_path.place(x=45, y=220)

        # todo sistemare la posizione di pain_label e fps_label ed eventualmente cambiare la loro visualizzazione
        self.pain_label = tk.Label(self.pain_frame)
        self.pain_label.config(font=('Times', 22))
        self.pain_label.place(x=10, y=50)
        self.pain_label.config(text="PAIN LEVEL: 999")

        self.fps_label = tk.Label(self.pain_frame)
        self.fps_label.config(font=('Times', 22))
        self.fps_label.place(x=10, y=100)

    def open_webcam(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.cap = cv2.VideoCapture(0)
        self.show_frame(self.start_time)

    def open_video(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.video_path = self.entry_video_path.get()
        self.cap = cv2.VideoCapture(self.video_path)
        self.show_frame(self.start_time)

    def show_frame(self, st):
        # start_time = time.time()
        ret, frame = self.cap.read()
        self.frame_count += 1
        plotting = True
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.image = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(self.image)
            if self.frame_count % 10 == 0:
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
                self.photo_g = ImageTk.PhotoImage(self.image_g.resize((500, 400)))
                self.plot_pain_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_g)
                self.plot_pain_canvas.image = self.photo_g

                plt.bar(au_list, au_row)
                plt.title('Action Units')
                plt.savefig('./graphs/au_bar_graph.png')
                plt.close()

                self.image_au = Image.open('./graphs/au_bar_graph.png')
                self.photo_au = ImageTk.PhotoImage(self.image_au.resize((500, 400)))
                self.plot_au_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_au)
                self.plot_au_canvas.image = self.photo_au

            elapsed_time = time.time() - st
            fps = self.frame_count / elapsed_time
            self.fps_label.config(text=f"FPS: {fps:.2f}")

            # Update the canvas with the new frame
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # Schedule the next frame update
            self.video_canvas.after(15, self.show_frame, self.start_time)

    """def open_image(self):
        # todo modificare predict se serve questo metodo
        self.image_path = self.entry_image_path.get()
        frame = cv2.imread(self.image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(frame)
        self.photo = ImageTk.PhotoImage(self.image)
        faces_fex = detector.detect_faces(frame, threshold=0.5)
        landmarks_fex = detector.detect_landmarks(frame, faces_fex)
        au_fex = detector.detect_aus(frame, landmarks_fex)
        df_au = pd.DataFrame(columns=au_r_list)
        indices = [8, 15, 18]
        au_row = np.delete(au_fex[0][0] * 5, indices)
        df_au.loc[len(df_au)] = au_row
        pain = model.predict(df_au)
        self.pain_label.config(text=f"PAIN LEVEL: {pain[0]:.2f}")

        plt.bar(['Pain Level'], pain)
        plt.ylim([0, 3])
        plt.savefig('pain_graph.png')
        plt.close()

        self.image_g = Image.open('../../Desktop/PainDetector/pain_graph.png')
        self.photo_g = ImageTk.PhotoImage(self.image_g.resize((500, 400)))
        self.plot_pain_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_g)
        self.plot_pain_canvas.image = self.photo_g

        plt.bar(au_list, au_row)
        plt.title('Action Units')
        plt.savefig('au_bar_graph.png')
        plt.close()

        self.image_au = Image.open('../../Desktop/PainDetector/au_bar_graph.png')
        self.photo_au = ImageTk.PhotoImage(self.image_au.resize((500, 400)))
        self.plot_au_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_au)
        self.plot_au_canvas.image = self.photo_au

        # Update the canvas with the new frame
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
"""

# model = None
# detector = None
model = keras.models.load_model('pain_model/modello.h5')
detector = Detector(face_model=face_model, landmark_model=landmark_model, au_model=au_model)
root = tk.Tk()
app = App(root)
root.mainloop()
