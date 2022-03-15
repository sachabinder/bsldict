"""
Demo on a single video short to search in the dictionary and display top 20 results.
Example usage:
    python main.py --input_path inputs/input.mp4
"""

import argparse     # pass argument into the shell (when calling the program "demo.py")
import math         # maths things
import os           # interact with the os
import pickle as pkl    # to save any Python object on a binary file (pickle.dump(myObj, myFile) to save, pickle.load(myFile) to load
from pathlib import Path    # to navigate through files tree

import cv2          # OpenCV : computer vision / image processing library
import numpy as np  # modules for basic tensors, scientific computation
import torch        # PyTorch, module for NN, models, passforward....
from sklearn.metrics import pairwise_distances  # module to compute distances
from tqdm import tqdm   # module to display loading bar
import sys

# see utils.py for the complete documentation
from utils import (
    load_model,
    load_rgb_video,
    prepare_input,
    sliding_windows,
    viz_similarities,
    frame_sampler
)

sys.path.append("..")

############# MAIN FUNCTION #############
def main(
    checkpoint_path: Path,
    bsldict_metadata_path: Path,
    input_path: Path,
    batch_size: int,
    num_top: int = 20,
    num_classes: int = 1064,
    num_in_frames: int = 16,
    fps: int = 25,
    embd_dim: int = 1024,
):
    """
    Run sign spotting demo:
    1) load the pre-extracted dictionary video features,
    2) load the pretrained model (BOBSL OR BSL1K),
    3) read the input video, preprocess it into samples (or sliding windows), extract its features,
    4) compare the input video features at every time step with all dictionary features

    The parameters are explained in the help value for each argument at the bottom of this code file.

    :param checkpoint_path: default `../models/i3d_mlp.pth.tar` should be used
    :param bsldict_metadata_path: default `../bsldict/bsldict_v1.pkl` should be used
    :param input_path: path to the continuous test video
    :param batch_size: how many sliding window clips to group when applying the model, this depends on the hardware resources, but doesn't change the results
    :param num_top: number of videos to display that most closely matches the input video
    :param num_classes: it depends on which model is loaded 1064 if BSL1K, 2281 if BOBSL
    :param num_in_frames: number of frames processed at a time by the model (I3D model is trained with 16 frames)
    :param fps: the frame rate at which to read the input video
    :param embd_dim: the video feature dimensionality, always 256 for the MLP model output or 1024 for I3D model output
    """

    #=================================================================================
    #=============================== Preprocessing ===================================
    #=================================================================================

    #=============================== BSLDict loading ===============================
    # check if the BSLDict is downloaded
    msg = "Please download the BSLDict metadata at bsldict/download_bsldict_metadata.sh"
    assert bsldict_metadata_path.exists(), msg

    print(f"Loading BSLDict data (words & features) from {bsldict_metadata_path}")

    with open(bsldict_metadata_path, "rb") as f:    #"rb" : open a file with reading (r) in binary (b)
        bsldict_metadata = pkl.load(f)  #load pkl : fichier binaire --> obj python

    # out of the model for dict videos
    dict_features = np.array(bsldict_metadata["videos"]["features"]["i3d_bsl1k"])

    dict_words = np.array(bsldict_metadata["videos"]["word"])  # dict words
    dict_video_urls = np.array(bsldict_metadata["videos"]["video_link_db"])  # URLS of corresponding videos


    #=============================== MODEL loading ===============================
    # Check if the model is downloaded
    msg = "Please download the pretrained model at models/download_models.sh"
    assert checkpoint_path.exists(), msg

    # Hardware configuration
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # check if there is a GPU on the computer (if not, move to CPU)
    print(f"Using {device} for computation.")

    # Loading the pretrained nural network
    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, device=device, num_classes=num_classes)


    #=============================== INPUT VIDEO loading ===============================
    # Load the continuous RGB video INPUT from a Path to a Torch Tensor
    rgb_orig = load_rgb_video(video_path=input_path, fps=fps)

    # Prepare: function of utils.py that resize to [256x256], center crop with [224x224], normalize colors in [-0.5;+ 0.5]
    rgb_input = prepare_input(rgb_orig)

    # Sampling rgb : rgb_samples (sampling of video) and t_mid (corresponding (middle) timestamp)
    # It produce a tensor of num_clips of clips (each of num_in_frames frame) in RGB with rgb_input[i,j] shapes
    rgb_samples, t_mid = frame_sampler(rgb=rgb_input, num_in_frames=num_in_frames)

    # Number of windows/clips
    num_clips = rgb_samples.shape[0]

    # Group clips into batches --> to feed the model batch by batch
    num_batches = math.ceil(num_clips / batch_size)


    # contain the output of the model for all batches
    continuous_features = np.empty((0, embd_dim), dtype=float)


    #=================================================================================
    #=============================== Feeding the model ===============================
    #=================================================================================

    for b in tqdm(range(num_batches)):
        inp = rgb_samples[b * batch_size : (b + 1) * batch_size] # input of the model
        inp = inp.to(device)    # move the torch Tensor on the GPU/CPU
        # Forward pass
        out = model(inp)
        # Append the solution to the continous_features
        continuous_features = np.append(
            continuous_features, out["embds"].cpu().detach().numpy()[:,:,0,0,0], axis=0
        )


    #===========================================================================
    #========================= compaire the output with the dict ===============
    #===========================================================================

    # Compute distance between continuous and dictionary features (Cosine distance <--> normalized dot product)
    dst = pairwise_distances(continuous_features, dict_features, metric="cosine")   # matrix of normalized dot product [num_clips, num_dict_video]
    # Convert from [-1,1] to [0, 1] similarity. Dimensionality: [ContinuousTimes x DictionaryVersions]
    sim = 1 - dst / 2

    # Associate the video to a single probability
    # avg_sim = np.mean(sim, axis=0)  # taking the average of all clips

    avg_sim = np.max(sim, axis=0)   # taking the proba max of all clips


    # sort the array and get indexes of the corresponding videos
    version_sorted_ix = np.flip(np.argsort(avg_sim, kind='quicksort'))

    for i in range(num_top):
        print(dict_words[version_sorted_ix[i]], " -- ", avg_sim[version_sorted_ix[i]], " -- ",
              dict_video_urls[version_sorted_ix[i]])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Helper script to run main.")
    p.add_argument(
        "--checkpoint_path",
        type=Path,
        default="../models/bsl1k_i3d.pth.tar",
        help="Path to i3d model.",
    )
    p.add_argument(
        "--bsldict_metadata_path",
        type=Path,
        default="../bsldict/bsldict_v2_i3d.pkl",
        help="Path to bsldict data",
    )
    p.add_argument(
        "--input_path",
        type=Path,
        default="inputs/input_apple.mp4",
        help="Path to input video.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Maximum number of clips to put in each batch",
    )
    p.add_argument(
        "--num_top",
        type=int,
        default=20,
        help="Number of videos to display that most closely matches the input video",
    )
    p.add_argument(
        "--num_classes",
        type=int,
        default=1064,
        help="It depends on which model is loaded 1064 if BSL1K, 2281 if BOBSL",
    )
    p.add_argument(
        "--num_in_frames",
        type=int,
        default=16,
        help="Number of frames processed at a time by the model",
    )
    p.add_argument(
        "--fps", type=int,
        default=25,
        help="The frame rate at which to read the video",
    )
    p.add_argument(
        "--embd_dim",
        type=int,
        default=1024,
        help="The feature dimensionality, 1024 for the i3d model output or 256 for the MLP model output.",
    )

    main(**vars(p.parse_args()))
