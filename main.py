from fastapi import FastAPI, File, UploadFile
# image classification
from fastapi.responses import FileResponse
import shutil
from torchvision import datasets, models, transforms
import torch
from PIL import Image
import joblib
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load('model/pred_face_direction_rf.pkl')

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# get method를 이용해 접근
@app.get('/')
def root():
    return {"message":"ok"}

# postman에서 post 방식으로 접근
@app.post('/sendimg')
def sendimg(file:UploadFile=File(...)):
    print(file.filename)
    path = f'files/{file.filename}'
    with open(path, 'w+b') as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(file.filename)
    
    image = cv2.imread(f'files/{file.filename}')
    # image = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
    with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
        result = holistic.process(image)

        face = result.face_landmarks.landmark
        face_list = []
        for temp in face:
            face_list.append([temp.x, temp.y, temp.z])
        face_row = list(np.array(face_list).flatten())

        X = pd.DataFrame([face_row])

        class_name = ['front', 'left', 'right']
        yhat = model.predict(X)[0]
        yhat = class_name[yhat]
        
    return {'result':yhat}