import cv2
import numpy as np
import os
import config

class FaceRecognizer:
    
    def __init__(self) -> None:
        self.faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(os.path.join(config.MODEL_DIR, 'trainer.yml'))
        self.userDir = os.path.join(config.DATA_DIR, config.USER_DIR)
        self.names = self.get_names()
        print(self.names)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def get_names(self):
        namesPath = os.path.join(self.userDir, "names.txt")
        with open(namesPath) as f:
            return f.read().split("\n")
    
    def recognize(self):
        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video widht
        cam.set(4, 480) # set video height
        # Define min window size to be recognized as a face
        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)
        while True:
            ret, img =cam.read()
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            faces = self.faceDetector.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
            )
            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                id, confidence = self.recognizer.predict(gray[y:y+h,x:x+w])
                
                id = (id // 100) + 1
                print(id)
                if (confidence < 100):
                    if id != 0:
                        id = self.names[id]
                        print(self.names)
                        print(id)
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                
                cv2.putText(
                            img, 
                            str(id), 
                            (x+5,y-5), 
                            self.font, 
                            1, 
                            (255,255,255), 
                            2
                        )
                cv2.putText(
                            img, 
                            str(confidence), 
                            (x+5,y+h-5), 
                            self.font, 
                            1, 
                            (255,255,0), 
                            1
                        )  
            cv2.imshow('camera',img) 
            k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()