"""
Dans ce fichier, il y a toutes les commandes que j'ai tapé dans le shell pour comprendre comment fonctionne le code
"""

# Etape 1 : comprendre comment le dictionnaire est importé
import os   # module qui permet de faire comme cd mais dans le shell Python
my_dir_path = "/mnt/FAACED9AACED51A5/Users/Sacha BINDER/linux_files/PReuch/PROG/bsldict/bsldict" # compléte avec ton chemin d'accés
os.chdir(my_dir_path)   # equivalent du cd dans le shell python

# Je recopie les commandes du fichier demo.py pour voir ce qui se passe
bsldict_metadata_path = "bsldict_v1.pkl"
f = open(bsldict_metadata_path, "rb")   # lecture du fichier
import pickle as pkl
bsldict_metadata = pkl.load(f)  # je stock mon fichier dans cette variable
f.close()   # je ferme le fichier

# je suis le cheminement du code
bsldict_metadata["videos"]["word"]
keyword = "apple"
import numpy as np
np.where(np.array(bsldict_metadata["videos"]["word"]) == keyword)
dict_ix = np.where(np.array(bsldict_metadata["videos"]["word"]) == keyword)[0]
dict_ix

dict_features = np.array(bsldict_metadata["videos"]["features"]["mlp"])[dict_ix]
dict_features

np.shape(dict_features)
dict_features.min()
dict_features.max()

dict_video_urls = np.array(bsldict_metadata["videos"]["video_link_db"])[dict_ix]
dict_video_urls


dict_youtube_ids = np.array(bsldict_metadata["videos"]["youtube_identifier_db"])[dict_ix]
dict_youtube_ids

#======================= sliding windows ===========================


import argparse     # pass argument into the shell (when calling the program "demo.py")
import math         # maths things
import pickle as pkl    # to save any Python object on a binary file (pickle.dump(myObj, myFile) to save, pickle.load(myFile) to load
from pathlib import Path    # to navigate through files tree

import cv2          # OpenCV : computer vision / image processing library
import torch        # PyTorch, module for NN, models, passforward....
from sklearn.metrics import pairwise_distances  # ML module ---> what is the diff with PyTorch...?
from tqdm import tqdm   # module to display loading bar


my_dir_path = "/mnt/FAACED9AACED51A5/Users/Sacha BINDER/linux_files/PReuch/PROG/bsldict/demo" # compléte avec ton chemin d'accés
os.chdir(my_dir_path)   # equivalent du cd dans le shell python

# see utils.py for the complete documentation
from utils import (
    load_model,
    load_rgb_video,
    prepare_input,
    sliding_windows,
    viz_similarities,
)


input_path = "/mnt/FAACED9AACED51A5/Users/Sacha BINDER/linux_files/PReuch/PROG/bsldict/demo/sample_data/input.mp4" # compléte avec ton chemin d'accés
fps = 25
rgb_orig = load_rgb_video(video_path=input_path, fps=fps, )



