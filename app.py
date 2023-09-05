import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import config  # Assuming you have a config module with your configuration
from providers.FaceRegistration import FaceRegistrar
from providers.FaceRecognizer import FaceRecognizer
# Import your FaceRecognizer and FaceRegistrar classes here

st.title("ATM Face Recognizer")

# Create an instance of the FaceRegistrar class

# Create a sidebar with options
option = st.sidebar.selectbox("Select an option", ("Create Account", "Recognize Face"))

if option == "Create Account":
    registrar = FaceRegistrar()
    st.subheader("Create an Account")
    name = st.text_input("Enter your name:")
    if name:
        account = {"owner_name": name}
        if st.button("Create Account"):
            registrar.train_for_face(account)
            st.success("Account created successfully!")

if option == "Recognize Face":
    recognizer = FaceRecognizer()
    st.subheader("Recognize Face")
    # Add a button to open the webcam
    if st.button("Recognize"):
        recognizer.recognize()