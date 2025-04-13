from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def extract_facial_features(image_path):

    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return 

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            landmarks.append((x, y))

        return image, np.array(landmarks)
    
def draw_landmark_rectangles(image, landmarks, indices):
    for idx in indices:
        x, y = landmarks[idx]
        top_left = (x - 25, y - 25)
        bottom_right = (x + 5, y + 5)
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2) 
    return image
    
def calculate_distance(features1, features2):
    features1_flat = features1.flatten()
    features2_flat = features2.flatten()
    return np.linalg.norm(features1_flat - features2_flat)

def generate_feature_vector(image_path):
    _, features = extract_facial_features(image_path)
    return features if features is not None else None

