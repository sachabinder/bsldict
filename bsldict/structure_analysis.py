"""
File to analyse the structure of dictionaries
"""

import pickle as pkl    # to save any Python object on a binary file (pickle.dump(myObj, myFile) to save, pickle.load(myFile) to load
import scipy.io

def dict_structure(data: dict, marker: str = "  "):
    for key, val in data.items():
        if type(val) is dict:
            print(marker, key, " -- ", type(val), " {")
            dict_structure(val, 2*marker)
            print("}")
        elif type(val) is not int:
            print(marker, key, " -- ", type(val))

if __name__ == "__main__":
    file_path1 = "bsldict_v2_i3d.pkl"

    with open(file_path1, "rb") as f:  # "rb" : open a file with reading (r) in binary (b)
        data1 = pkl.load(f)  # load pkl : fichier binaire --> obj python

    dict_structure(data1)



