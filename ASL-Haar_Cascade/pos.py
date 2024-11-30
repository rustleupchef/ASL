import cv2
import uuid
from time import sleep
import os

PATH = "Positives/"
labels = ["hello", "ly", "no", "thanks", "yes"]
NUM_IMGS = 85

video = cv2.VideoCapture(0)

for label in labels:
    print(f"Now working on {label}")
    input("Continue?:")
    for i in range(NUM_IMGS):
        _,frame = video.read()
        cv2.imwrite(os.path.join(PATH, label, f"{label}{uuid.uuid1()}.jpg"), frame)
        cv2.imshow("frame", frame)
        sleep(3)
        if cv2.waitKey(1) == 27:
            break