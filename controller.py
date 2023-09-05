from providers.FaceRecognizer import FaceRecognizer
# from providers.FaceRegistration import FaceRegistrar
# register = FaceRegistrar()

account = {
    "owner_name": "Siddhant",
    "id": "1"
}
# register.train_for_face(account)
recognizer = FaceRecognizer()
recognizer.recognize()