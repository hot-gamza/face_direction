import mediapipe as mp
import cv2
import numpy as np
from os import path
import csv

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # 캡처된 프레임이 변경되지 못하게 copy하여 사용하거나 image.flags.writeable를 false로 지정함

        result = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, 
                                  result.face_landmarks,
                                  mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1))
        
        try:
            # print(result.face_landmarks.landmark)
            face = result.face_landmarks.landmark
            # columns : x,y,z,x,y,z...
            face_coor = []
            for temp in face:
                face_coor.append([temp.x, temp.y, temp.z]) # 468개의 포인트가 저장 -> 468개의 리스트에 [x,y,x]로 들어감
            # face_list를 1차원으로 펼쳐 csv로 저장
            face_coordinates = list(np.array(face_coor).flatten())

            if path.isfile('facedirection.csv') == False:
                landmarks = ['class']
                for val in range(1, len(face)+1):
                    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)] # ['class', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2'...]

                with open('facedirection.csv', mode='w', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(landmarks) # csv 첫줄에 landmarks 값 저장
                f.close()
            else:
                if cv2.waitKey(1) & 0xFF == ord('1'):
                    face_coordinates.insert(0, 'front')
                    with open('facedirection.csv', mode='a', newline='') as f: # mode를 a로 지정해 작성 누적
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(face_coordinates)
                elif cv2.waitKey(1) & 0xFF == ord('2'):
                    face_coordinates.insert(0, 'left')
                    with open('facedirection.csv', mode='a', newline='') as f: 
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(face_coordinates)
                elif cv2.waitKey(1) & 0xFF == ord('3'):
                    face_coordinates.insert(0, 'right')
                    with open('facedirection.csv', mode='a', newline='') as f: 
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(face_coordinates)

        except:
            pass

        cv2.imshow('face', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()