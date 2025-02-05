import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import tasks
import render
import gfxmath
import cv2
import time
import numpy as np
from Entities import Entity
from tkinter import Tk, Button, Label, Frame, Canvas
from PIL import Image, ImageTk

import trimesh

def normalize_model(file_path, scale, translation, rotation):
    # Charger le modèle
    model = trimesh.load(file_path)
    if isinstance(model, trimesh.Scene):
        model = list(model.geometry.values())[0]

    # Appliquer l'échelle
    model.apply_scale(scale)

    # Appliquer une rotation (rotation en degrés autour des axes X, Y, Z)
    rotation_matrix = trimesh.transformations.euler_matrix(
        rotation[0], rotation[1], rotation[2], axes='sxyz'
    )
    model.apply_transform(rotation_matrix)

    # Appliquer la translation
    translation_matrix = trimesh.transformations.translation_matrix(translation)
    model.apply_transform(translation_matrix)

    # Sauvegarder le modèle transformé
    output_path = file_path.replace(".obj", "_normalized.obj")
    model.export(output_path)
    print(f"Modèle normalisé enregistré sous {output_path}")

# Paramètres communs à appliquer à tous les modèles
common_scale = 8  # Ajustez la taille
common_translation = [-0.5, 2, 4]  # Translation x, y, z
common_rotation = [0, 180, 0]  # Rotation x, y, z (en degrés)

# Normaliser chaque modèle
normalize_model('./objs/sunglass/model.obj', common_scale, common_translation, common_rotation)
normalize_model('./objs/sunglass/Glasses.obj', common_scale, common_translation, common_rotation)
normalize_model('./objs/sunglass/model3.obj', common_scale, common_translation, common_rotation)

class VirtualGlassesApp:
    def __init__(self, root):
        self.root = root
        self.facePose = None
        self.width = 640
        self.height = 480
        self.sunglassEnt = None
        self.sunglassOffset = None

        self.setup_mediapipe()
        self.setup_renderer()

        # Interface Tkinter
        self.create_interface()

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.update_frame()

    def setup_mediapipe(self):
        face_model_path = "./Models/face_landmarker_v2_with_blendshapes.task"
        face_options = vision.FaceLandmarkerOptions(
            base_options=tasks.BaseOptions(model_asset_path=face_model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            output_facial_transformation_matrixes=True,
            result_callback=self.get_face_pose_result,
        )
        self.faceLandmarker = vision.FaceLandmarker.create_from_options(face_options)

    def setup_renderer(self):
        render.SetupRenderer(self.width, self.height)
        render.SetupScene()
        self.matPsudoCam, self.distortionPsudoCam = gfxmath.GetPsudoCamera(self.width, self.height)

    def load_sunglasses1(self, model_path):
        self.sunglassOffset = gfxmath.makePose(translation=[-1.5, 1.5, 4], rotation=[0, 180, 0], scale=[9, 8, 8])
        self.sunglassEnt = Entity(model_path, self.sunglassOffset)
        render.LoadObj(self.sunglassEnt)
        print(f"Lunettes chargées : {model_path}")

    def load_sunglasses2(self, model_path):
        self.sunglassOffset = gfxmath.makePose(translation=[0, 0, 6], rotation=[0, 0, 0], scale=[85, 85, 85])
        self.sunglassEnt = Entity(model_path, self.sunglassOffset)
        render.LoadObj(self.sunglassEnt)
        print(f"Lunettes chargées : {model_path}")

    def load_sunglasses3(self, model_path):
        self.sunglassOffset = gfxmath.makePose(translation=[0, 2, 0], rotation=[0, 180, 0], scale=[1.7, 1.7, 1.7])
        self.sunglassEnt = Entity(model_path, self.sunglassOffset)
        render.LoadObj(self.sunglassEnt)
        print(f"Lunettes chargées : {model_path}")

    def load_sunglasses(self, model_path):
        if model_path.endswith(".glb"):
            self.sunglassOffset = gfxmath.makePose(translation=[-1.5, 1.5, 4], rotation=[0, 180, 0], scale=[9, 8, 8])
        else:
            self.sunglassOffset = gfxmath.makePose(translation=[-0.5, 2, 0], rotation=[0, 180, 0], scale=[1.7, 1.7, 1.7])
    
        self.sunglassEnt = Entity(model_path, self.sunglassOffset)
        render.LoadObj(self.sunglassEnt)  # Assurez-vous que render supporte GLB
        print(f"Lunettes chargées : {model_path}")


    def get_face_pose_result(self, result: vision.FaceLandmarkerOptions, output_image: mp.Image, timestamp_ms: int):
        if len(result.facial_transformation_matrixes) > 0:
            self.facePose = result.facial_transformation_matrixes[0]

    def composite(self, image, overlay, ignore_color=[0, 0, 0]):
        ignore_color = np.full(overlay.shape, ignore_color)
        mask = ~(overlay == ignore_color).all(-1)
        if image.shape[:2] != overlay.shape[:2]:
            overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))
        out = image.copy()
        out[mask] = image[mask] * 0.5 + overlay[mask] * 0.5
        return out

    def update_frame(self):
        if not self.running:
            return

        ret, image = self.cap.read()
        if not ret:
            print("Erreur : Impossible de lire le flux vidéo.")
            self.root.after(10, self.update_frame)
            return

        # Miroir et redimensionner l'image
        image = cv2.flip(image, 1)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # Convertir en format Mediapipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        timeStampMs = current_milli_time()

        # Effectuer la détection des landmarks
        self.faceLandmarker.detect_async(mp_image, timeStampMs)

        if self.facePose is not None and self.sunglassEnt is not None:
            self.sunglassEnt.SetTransform(np.dot(self.facePose, self.sunglassOffset))
            render.Draw(self.sunglassEnt)

        # Fusionner la vidéo et le rendu
        outImage = self.composite(image, render.colorBuffer, [0, 0, 0])
        render.Clear()

        # Convertir pour afficher avec Tkinter
        outImage = cv2.cvtColor(outImage, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(outImage))
        self.video_canvas.create_image(0, 0, anchor="nw", image=img)
        self.video_canvas.image = img

        # Mettre à jour l'image après un petit délai
        self.root.after(10, self.update_frame)

    def create_interface(self):
        # Diviser la fenêtre en deux parties
        self.video_canvas = Canvas(self.root, width=self.width, height=self.height)
        self.video_canvas.grid(row=0, column=0)

        button_frame = Frame(self.root, width=200, height=self.height, bg="lightgray")
        button_frame.grid(row=0, column=1, sticky="ns")

        Label(button_frame, text="Changer les Lunettes", bg="lightgray").pack(pady=10)

        # Ajouter les boutons pour changer les lunettes
        Button(button_frame, text="Lunettes 1", command=lambda: self.load_sunglasses1('./objs/sunglass/model.obj')).pack(pady=10)
        Button(button_frame, text="Lunettes 2", command=lambda: self.load_sunglasses2('./objs/sunglass/glasses.obj')).pack(pady=10)
        Button(button_frame, text="Lunettes 3", command=lambda: self.load_sunglasses3('./objs/sunglass/model3.obj')).pack(pady=10)


    def close_app(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


def current_milli_time():
    return round(time.time() * 1000)


if __name__ == "__main__":
    root = Tk()
    root.title("Lunettes Virtuelles 3D")
    app = VirtualGlassesApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()
