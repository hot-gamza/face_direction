import cv2
import mediapipe as mp
import numpy as np
import joblib # pkl 모델 사용

import pandas as pd
import warnings

warnings.filterwarnings('ignore')

model = joblib.load('pred_face_direction_lr.pkl')

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        result = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, 
                                  result.face_landmarks,
                                  mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1))
        
        try:
            face = result.face_landmarks.landmark
            face_list = []
            for temp in face:
                face_list.append([temp.x, temp.y, temp.z])
            face_row = list(np.array(face_list).flatten())

            X = pd.DataFrame([face_row])

            class_name = ['front', 'left', 'right']
            yhat = model.predict(X)[0]
            yhat = class_name[yhat]

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, yhat, (30,30), font, 1, (255,0,0), 2)

        except:
            pass
        
        cv2.imshow('face', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
