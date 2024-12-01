import PIL.Image as img
import google.generativeai as genai
import cv2
import os
from dotenv import load_dotenv
import torch
import numpy as np
from gtts import gTTS
import threading
import sys

load_dotenv()

isSpeaking = False

def grabCommand() -> str:
    if sys.platform == "linux": 
        return "aplay sound.wav"
    return "ffplay -nodisp -autoexit sound.wav" if sys.platform == "win32" else "afplay sound.wav"

def speak(text: str) -> None:
    global isSpeaking
    print(isSpeaking)
    if isSpeaking == True: return
    isSpeaking = True
    gTTS(text, lang='en').save('sound.wav')
    os.system(grabCommand())
    isSpeaking = False

def setVersion(isRealTime: bool) -> None:
    global model
    if isRealTime:
        model = torch.hub.load(repo_or_dir="ultralytics/yolov5", model="custom", path="yolov5/runs/train/exp7/weights/best.pt", force_reload=True)
        model.conf = 0.65
        return
    genai.configure(api_key=os.getenv("API_KEY"))
    with open("instructions.txt", 'r') as file:
        model = genai.GenerativeModel("gemini-1.5-flash-002", system_instruction=file.read())
        file.close()
        pass   


def main() -> None:
    threads: list[threading.Thread] = []
    isRealTime: bool = True
    while True:
        inputText = input("Realtime - R or Gemini - G: ").upper()
        if inputText == "R" or inputText == "G":
            isRealTime = (inputText == "R")
            setVersion(isRealTime)
            break
        print("Please enter either R for realtime or G for gemini")
    
    video = cv2.VideoCapture(0)

    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break
        _, frame = video.read()

        if not isRealTime:
            cv2.imshow("frame", frame)
            if key == ord('f') :
                cv2.imwrite("output.png", frame)
                image = img.open("output.png")
                response = model.generate_content([image, "What ASL sign is made in this image if any?"]).text
                thread = threading.Thread(target=speak, args=[response])
                threads.append(thread)
                thread.start()
            continue
        
        
        results = model(frame)
        detections = results.pandas().xyxy[0]['name'].tolist()
        for detection in detections:
            thread = threading.Thread(target=speak, args=[detection])
            threads.append(thread)
            thread.start()
        cv2.imshow("frame", np.squeeze(results.render()))

    for thread in threads:
        thread.join()
    video.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()