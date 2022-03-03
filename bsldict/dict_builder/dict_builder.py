"""
File that takes info videos and bsldict features of models to build one only bsldict pkl file
"""

import  numpy as np
from scipy.io import loadmat
import pickle as pkl

print("Loading files...")


# files path
vInfo_file_path = "info_v00_20.03.30.pkl"

mat_bobsl_path = 'bobsl-bsldict_features.mat'
mat_bsl1k_path = 'bsl1k-bsldict_features.mat'

out_file_name = "../bsldict_v2_i3d.pkl"


# files loading
with open(vInfo_file_path, "rb") as f:  # "rb" : open a file with reading (r) in binary (b)
    data = pkl.load(f)  # load pkl : fichier binaire --> obj python

mat_bobsl = loadmat(mat_bobsl_path)
mat_bsl1k = loadmat(mat_bsl1k_path)

print("Files loaded")


# extract video names
names_bobsl = [elt.replace(" ", "") for elt in mat_bobsl["video_names"]]
names_bsl1k = [elt.replace(" ", "") for elt in mat_bsl1k["video_names"]]

names = data["videos"]["name"]

# extract video features
features_bobsl = [list(elt) for elt in mat_bobsl["vid_features"]]
features_bsl1k = [list(elt) for elt in mat_bsl1k["vid_features"]]

print("Data extracted")

# create the new dictionary
assert names_bobsl == names, f"The order of videos in {vInfo_file_path} and {mat_bobsl_path} is not maching"
assert names_bobsl == names, f"The order of videos in {vInfo_file_path} and {mat_bsl1k} is not maching"

data["videos"]["features"] = {"i3d_bobsl":features_bobsl, "i3d_bsl1k":features_bobsl}
data["videos"]["youtube_identifier_db"] = [None]*len(names)


# youtube id extraction
ytdl_method_ix = np.where(np.array(data["videos"]["download_method_db"]) == "youtube-dl")[0]

yt_links = np.array(data["videos"]["video_link_db"])[ytdl_method_ix]
yt_id = [elt.split("/")[4].split("?")[0] for elt in yt_links]

for i, id in zip(ytdl_method_ix, yt_id):
    data["videos"]["youtube_identifier_db"][i] = id


# saving pkl file

print(f"Saving data in {out_file_name}...")

with open(out_file_name, "wb") as f:  # "wb" : open a file with write  (r) in binary (b)
    pkl.dump(data, f)

print(f"{out_file_name} saved !")









