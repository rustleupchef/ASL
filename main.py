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
torch.set_printoptions(threshold=0)
isSpeaking = False
dictionary: dict = {}

def grabCommand() -> str:
    if sys.platform == "linux": 
        return "aplay sound.wav"
    return "ffplay -nodisp -autoexit sound.wav" if sys.platform == "win32" else "afplay sound.wav"

def speak(text: str = None, image = None) -> None:
    if text is None:
        text = gmodel.generate_content([image, "What ASL sign is made in this image if any?"]).text
    global isSpeaking
    if isSpeaking == True: return
    isSpeaking = True
    gTTS(text, lang='en').save('sound.wav')
    os.system(grabCommand())
    isSpeaking = False

def setVersion() -> None:
    global dictionary
    with open("Dictionary.csv", 'r') as file:
        text = file.read().split('\n')
        file.close()
    for line in text:
        items = line.split(',')
        dictionary[str(items[0:-1])] = items[-1]
    print(dictionary)
    global model
    model = torch.hub.load(repo_or_dir="yolov5/", 
                            model="custom", 
                            path="models/ASL.pt", 
                            source="local", 
                            force_reload=True)
    model.conf = 0.55
    genai.configure(api_key=os.getenv("API_KEY"))
    with open("instructions.txt", 'r') as file:
        global gmodel
        gmodel = genai.GenerativeModel("gemini-1.5-flash-002", system_instruction=file.read())
        file.close()
        pass   

def main() -> None:
    threads: list[threading.Thread] = []
    setVersion()
    isChain: bool = False
    isFirstly: bool = True
    isEnded: bool = False
    chain: list[str] = []

    video = cv2.VideoCapture(0)
    local_model = model

    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break
        _, frame = video.read()
        frame = cv2.resize(frame, (640, 480))
        if key == ord('f') :
            cv2.imwrite("output.png", frame)
            image = img.open("output.png")
            thread = threading.Thread(target=speak, args=[None,image])
            threads.append(thread)
            thread.start()
        
        results = local_model(frame, size=640)
        detections = results.pandas().xyxy[0]['name'].tolist()
        for detection in detections:
            if detection == "ly":
                if isEnded: continue
                if isFirstly:
                    isChain = not isChain
                    if not isChain:
                        listStr = str(chain)
                        if listStr in dictionary.keys():
                            thread = threading.Thread(target=speak, args=[dictionary[listStr]])
                            threads.append(thread)
                            thread.start()
                            chain.clear()
                        isFirstly = True
                        isEnded = True
                    else:
                        isFirstly = False
                break
            elif isFirstly:
                isEnded = False
            if isChain:
                if len(chain) == 0 or detection != chain[-1]: 
                    chain.append(detection)
                    isFirstly = True
                break
            thread = threading.Thread(target=speak, args=[detection])
            threads.append(thread)
            thread.start()
            isFirstly = True
        cv2.imshow("frame", np.squeeze(results.render()))

    for thread in threads:
        thread.join()
    video.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()