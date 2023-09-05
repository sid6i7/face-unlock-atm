import numpy as np
import cv2
import os
import config
import utils
from PIL import Image

class FaceRegistrar:

    def __init__(self) -> None:
        self.faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.userDir = os.path.join(config.DATA_DIR, config.USER_DIR)

    def get_webcam(self):
        cap = cv2.VideoCapture(0)
        cap.set(3,640) # set Width
        cap.set(4,480) # set Height
        while(True):
            ret, frame = cap.read()

            cv2.imshow('frame', frame)

            k = cv2.waitKey(30) & 0xff
            if k == 27: # press 'ESC' to quit
                break
        cap.release()
        cv2.destroyAllWindows()
    
    def detect_face(self):
        cap = cv2.VideoCapture(0)
        cap.set(3,640) # set Width
        cap.set(4,480) # set Height
        while True:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.faceDetector.detectMultiScale(
                gray,     
                scaleFactor=1.2,
                minNeighbors=5,     
                minSize=(20, 20)
            )
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]  
            cv2.imshow('video',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27: # press 'ESC' to quit
                break
        cap.release()
        cv2.destroyAllWindows()
    
    def __register_face(self, account):
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video width
        cam.set(4, 480) # set video height
        count = 0
        accountDir = os.path.join(self.userDir, account['owner_name'])
        utils.create_dir(accountDir)
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.faceDetector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                count += 1
                # Save the captured image into the datasets folder
                cv2.imwrite(os.path.join(accountDir, str(count) + ".jpg"), gray[y:y+h,x:x+w])
                cv2.imshow('image', img)
            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 100: # Take 30 face sample and stop video
                break
        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

    def __get_images_and_labels(self, path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') # grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[0])
            faces = self.faceDetector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        return faceSamples,ids
    
    def train_for_face(self, account):
        self.__register_face(account)
        print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        accountDir = os.path.join(self.userDir, account['owner_name'])
        faces,ids = self.__get_images_and_labels(accountDir)
        self.recognizer.train(faces, np.array(ids))
        modelPath = os.path.join(config.MODEL_DIR, 'trainer.yml')
        self.recognizer.write(modelPath) 
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))