import pickle as pkl    # to save any Python object on a binary file (pickle.dump(myObj, myFile) to save, pickle.load(myFile) to load
from pathlib import Path    # to navigate through files tree

import cv2          # OpenCV : computer vision / image processing library
import numpy as np  # modules for basic tensors, scientific computation

file_path = "info_v00_20.03.30.pkl"

with open(file_path, "rb") as f:  # "rb" : open a file with reading (r) in binary (b)
    data = pkl.load(f)  # load pkl : fichier binaire --> obj python

def dict_structure(data: dict, marker: str = "  "):
    for key, val in data.items():
        if type(val) is dict:
            print(marker, key, " -- ", type(val), " {")
            dict_structure(val, 2*marker)
            print("}")
        elif type(val) is not int:
            print(marker, key, " -- ", type(val))

if __name__ == "__main__":
    file_path = "info_v00_20.03.30.pkl"

    with open(file_path, "rb") as f:  # "rb" : open a file with reading (r) in binary (b)
        data = pkl.load(f)  # load pkl : fichier binaire --> obj python

    dict_structure(data)
