import cv2 as cv

hello = cv.CascadeClassifier('cascade/cascade.xml')
video = cv.VideoCapture(0)
while cv.waitKey(1) != 27:
    ret,frame = video.read()
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rectangles = hello.detectMultiScale(grayFrame, scaleFactor=1.3, minNeighbors=7)
    if len(rectangles) > 0:
        for (x, y, w, h) in rectangles:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    cv.imshow("frame", frame)


video.release()
cv.destroyAllWindows()