import cv2
import os

P = "Positives/"
N = "Negatives/"

labels = ["hello", "ly", "no", "thanks", "yes"]

with open("neg.txt", 'w') as file:
    for name in os.listdir(N):
        file.write(f"{N}{name}\n")
    file.close()
    pass