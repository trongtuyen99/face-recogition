import numpy as np
import os
import cv2
import time
from keras.optimizers import Adam
from keras.models import model_from_json

# label = ['11', '189', '204', '230', '244', '274', '4', '434', '459', '462',
#        '478', '482', '509', '572', '576', '594', '660', '666', '680',
#        '720', '731', '760', '777', '827', '94', '959', '994', '997',
#        '998']

mapping_name = {0: 'khong ro ten', 1: 'pewpew', 2: 'trung ruoi', 3: 'vanh leg', 4: 'quang thang', 5: 'dinh la thang ko ro',
                6: 'b girl ko ro', 7: 'ms thy', 8: 'huynh duc ko ro', 9: 'bau duc', 10: 'bui tien dung',
                11: 'bui nhat anh', 12: 'vinh rau', 13: 'virus', 14: 'bac j do', 15: 'quynh kool', 16: 'black bi',
                17: 'viet johan', 18: 'chi xinh gai', 19: 'minh tit', 20: 'huynh phuong',
                21: 'nguyen van duc', 22: 'mac van khoa', 23: 'phan manh quynh', 24: 'chi xinh dep', 25: 'duy manh',
                26: 'chu tich vietjet air', 27: 'Den vau', 28: 'van dung'}

path_test_video = "./test_video_denvau.mp4"

fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0, 255, 0)
fontcolor1 = (0, 0, 255)

face_detect = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

json_file = open('./model_92.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./model_92.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

video_capture = cv2.VideoCapture(0)
# video_capture = cv2.VideoCapture(path_test_video)

while True: # video_capture.isOpened()
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(100, 100),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        face = gray[x:x+w, y:y+h]/255
        face = cv2.resize(face, (128, 128))
        score = loaded_model.predict(face.reshape(1, 128, 128, 1))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        idx = np.argmax(score)
        if np.max(score) > 0.3:
            cv2.putText(frame, "Hello " + mapping_name[idx], (x, y + h + 30), fontface, fontscale, fontcolor, 2)
        else:
            cv2.putText(frame, "Unknown", (x, y + h + 30), fontface, fontscale, fontcolor, 2)
    cv2.imshow("face detection", frame)
    time.sleep(0.05)
    key = cv2.waitKey(1) & 0xFF # press esc to exit
    if key == 27:
        break
video_capture.release()
cv2.destroyAllWindows()



