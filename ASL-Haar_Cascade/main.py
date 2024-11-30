import cv2 as cv
import os

hello = cv.CascadeClassifier('Cascades/hello_cascade.xml')

if input("Y/N? - ").lower() == "y":
    video = cv.VideoCapture(0)
    while cv.waitKey(1) != 27:
        ret,frame = video.read()
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        rectangles = hello.detectMultiScale(grayFrame, scaleFactor=1.2)
        if len(rectangles) > 0:
            for (x, y, w, h) in rectangles:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        cv.imshow("frame", frame)
    
    video.release()
else:
    names = os.listdir("Positives/hello/")
    i = 0
    while True:
        key = cv.waitKey(1)
        frame = cv.imread(f"Positives/hello/{names[i]}")
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        rectangles = hello.detectMultiScale(grayFrame, scaleFactor=1.1)
        if len(rectangles) > 0:
            for (x, y, w, h) in rectangles:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        cv.imshow("frame", frame)
        if key == 27: break
        if key == ord('q'):
            i = (i + 1) % len(names)
cv.destroyAllWindows()
