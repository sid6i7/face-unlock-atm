import os

def create_dir(dirName):
    os.makedirs(dirName, exist_ok=True)