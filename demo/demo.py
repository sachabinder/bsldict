"""
Demo on a single video input and a keyword to search in the dictionary.
Example usage:
    python demo.py --input_path sample_data/input.mp4 --keyword apple --output_path sample_data/output_apple.mp4
    python demo.py --input_path sample_data/input.mp4 --keyword garden --output_path sample_data/output_garden.mp4
    python demo.py --input_path sample_data/input.mp4 --keyword tree --output_path sample_data/output_tree.mp4
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


# see utils.py for the complete documentation
from utils import (
    load_model,
    load_rgb_video,
    prepare_input,
    sliding_windows,
    viz_similarities,
)


############# MAIN FUNCTION #############
def main(
    checkpoint_path: Path,
    bsldict_metadata_path: Path,
    keyword: str,
    input_path: Path,
    viz: bool,
    output_path: Path,
    viz_with_dict: bool,
    gen_gif: bool,
    similarity_thres: float,
    batch_size: int,
    stride: int = 1,
    num_in_frames: int = 16,
    fps: int = 25,
    mlp: bool = 1,
    embd_dim: int = 256,
):
    """
    Run sign spotting demo:
    1) load the pre-extracted dictionary video features,
    2) load the pretrained model,
    3) read the input video, preprocess it into sliding windows, extract its features,
    4) compare the input video features at every time step with the dictionary features
        corresponding to the keyword
    5) select the location with the highest similarity, if above a threshold, as spotting,
    6) (optional) visualize the similarity plots for each dictionary version corresponding to the keyword,
        save the visualization as video (and gif).
    
    The parameters are explained in the help value for each argument at the bottom of this code file.

    :param checkpoint_path: default `../models/i3d_mlp.pth.tar` should be used
    :param bsldict_metadata_path: default `../bsldict/bsldict_v1.pkl` should be used
    :param keyword: a search keyword, by default "apple", should exist in the dictionary
    :param input_path: path to the continuous test video
    :param viz: if 1, saves .mp4 visualization video
    :param output_path: path to the .mp4 visualization (used if viz)
    :param viz_with_dict: if 1, adds the dictionary frames to the visualization (downloads dictionary videos and takes middle frames)
    :param similarity_thres: similarity threshold that determines when a spotting occurs, 0.7 is observed to be a good value
    :param batch_size: how many sliding window clips to group when applying the model, this depends on the hardware resources, but doesn't change the results
    :param stride: how many frames to stride when applying sliding windows to the input video (1 obtains best performance)
    :param num_in_frames: number of frames processed at a time by the model (I3D model is trained with 16 frames)
    :param fps: the frame rate at which to read the input video
    :param mlp: if 1 based on mlp output else i3d
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


    # check if the keyword is in the BSLDict
    msg = f"Search item '{keyword}' does not exist in the sign dictionary."
    assert keyword in bsldict_metadata["words"], msg

    # Find dictionary videos whose sign corresponds to the searched key
    dict_ix = np.where(np.array(bsldict_metadata["videos"]["word"]) == keyword)[0]  # indexes of videos correspondig to the keyword (in bsldict)
    print(f"Found {len(dict_ix)} dictionary videos for the keyword {keyword}.")

    # out of the model for dict videos
    if mlp:
        print("Using the I3D + MLP model")
        dict_features = np.array(bsldict_metadata["videos"]["features"]["mlp"])[dict_ix]
    else:
        print("Using the only I3D model")
        dict_features = np.array(bsldict_metadata["videos"]["features"]["i3d"])[dict_ix]

    dict_video_urls = np.array(bsldict_metadata["videos"]["video_link_db"])[dict_ix]    # URLS of corresponding videos
    dict_youtube_ids = np.array(bsldict_metadata["videos"]["youtube_identifier_db"])[dict_ix]   # Some datas are on YT

    # Print found videos
    for vi, v in enumerate(dict_video_urls):
        print(f"v{vi + 1}: {v}")


    #=============================== MODEL loading ===============================
    # Check if the model is downloaded
    msg = "Please download the pretrained model at models/download_models.sh"
    assert checkpoint_path.exists(), msg

    # Loading the pretrained nural network
    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path = checkpoint_path, return_i3d_embds = not mlp) # function from utils.py that allow to load a pre-trained model (torch.nn)

    # Hardware configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # check if there is a GPU on the computer (if not, move to CPU)
    print(f"Moving model to {device}")
    model = model.to(device)    # Torch function that moving the model on GPU (if there is) else on CPU


    #=============================== INPUT VIDEO loading ===============================
    # Load the continuous RGB video INPUT from a Path to a Torch Tensor
    rgb_orig = load_rgb_video(video_path=input_path, fps=fps)

    # Prepare: function of utils.py that resize to [256x256], center crop with [224x224], normalize colors in [-0.5;+ 0.5]
    rgb_input = prepare_input(rgb_orig)

    # Sliding window : rgb_slides (sliding windows) and t_mid (corresponding (middle) timestamp)
    # It produce a tensor of num_clips of clips (each of num_in_frames frame) in RGB with rgb_input[i,j] shapes
    rgb_slides, t_mid = sliding_windows(rgb=rgb_input, stride=stride, num_in_frames=num_in_frames)


    # Number of windows/clips
    num_clips = rgb_slides.shape[0]

    # Group clips into batches --> to feed the model batch by batch
    num_batches = math.ceil(num_clips / batch_size)


    # contain the output of the model for all batches
    continuous_features = np.empty((0, embd_dim), dtype=float)


    #=================================================================================
    #=============================== Feeding the model ===============================
    #=================================================================================
    for b in tqdm(range(num_batches)):
        inp = rgb_slides[b * batch_size : (b + 1) * batch_size] # input of the model
        inp = inp.to(device)    # move the torch Tensor on the GPU/CPU
        # Forward pass
        out = model(inp)
        # Append the solution to the continous_features
        continuous_features = np.append(
            continuous_features, out["embds"].cpu().detach().numpy(), axis=0
        )


    #===========================================================================
    #========================= compaire the output with the dict ===============
    #===========================================================================

    # Compute distance between continuous and dictionary features (Cosine distance <--> normalized dot product)
    dst = pairwise_distances(continuous_features, dict_features, metric="cosine")   # matrix of normalized dot product [num_clips, num_dict_video]
    # Convert from [-1,1] to [0, 1] similarity. Dimensionality: [ContinuousTimes x DictionaryVersions]
    sim = 1 - dst / 2

    # Time where the similarity peaks
    peak_ix = sim.max(axis=1).argmax()
    # Dictionary version which responds with highest similarity
    version_ix = sim.argmax(axis=1)[peak_ix]
    max_sim = sim[peak_ix, version_ix]

    # If above a threhsold: spotted
    if max_sim >= similarity_thres:
        print(
            f"Sign '{keyword}' spotted at timeframe {peak_ix} "
            f"with similarity {max_sim:.2f} for the dictionary version {version_ix + 1}."
        )
    else:
        print(f"Sign {keyword} not spotted.")




    #===========================================================
    #========================= exporting results ===============
    #===========================================================

    # Visualize similarity plot
    if viz:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        # Save visualization video
        viz_similarities(
            rgb=rgb_orig,
            t_mid=t_mid,
            sim=sim,
            similarity_thres=similarity_thres,
            keyword=keyword,
            output_path=output_path,
            viz_with_dict=viz_with_dict,
            dict_video_links=(dict_video_urls, dict_youtube_ids),
            num_in_frames=num_in_frames
        )
        # Generate a gif
        if gen_gif:
            gif_path = output_path.with_suffix(".gif")
            cmd = f"ffmpeg -loglevel panic -y -i {output_path} -f gif {gif_path}"
            print(f"Generating gif of output at {gif_path}")
            os.system(cmd)



if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Helper script to run demo.")
    p.add_argument(
        "--checkpoint_path",
        type=Path,
        default="../models/i3d_mlp.pth.tar",
        help="Path to combined i3d_mlp model.",
    )
    p.add_argument(
        "--bsldict_metadata_path",
        type=Path,
        default="../bsldict/bsldict_v1.pkl",
        help="Path to bsldict data",
    )
    p.add_argument(
        "--keyword",
        type=str,
        default="apple",
        help="An item in the sign language dictionary",
    )
    p.add_argument(
        "--input_path",
        type=Path,
        default="sample_data/inputs/input_apple.mp4",
        help="Path to test video.",
    )
    p.add_argument(
        "--viz", type=int,
        default=1,
        help="Whether to visualize the predictions."
    )
    p.add_argument(
        "--output_path",
        type=Path,
        default="sample_data/output.mp4",
        help="Path to save viz (if viz=1).",
    )
    p.add_argument(
        "--viz_with_dict",
        type=bool,
        default=1,
        help="Whether to download dictionary videos for visualization.",
    )
    p.add_argument(
        "--gen_gif",
        type=bool,
        default=1,
        help="if true, also generate a .gif file of the output",
    )
    p.add_argument(
        "--similarity_thres",
        type=float,
        default=0.7,
        help="Only show matches above certain similarity threshold [0, 1]",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Maximum number of clips to put in each batch",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Number of frames to stride when sliding window in time.",
    )
    p.add_argument(
        "--num_in_frames",
        type=int,
        default=16,
        help="Number of frames processed at a time by the model",
    )
    p.add_argument(
        "--fps", type=int, default=25, help="The frame rate at which to read the video",
    )
    p.add_argument(
        "--mlp",
        type=bool,
        default=0,
        help="if true, using the mlp output for computation esle i3d.",
    )
    p.add_argument(
        "--embd_dim",
        type=int,
        default=1024,
        help="The feature dimensionality, 1024 for the i3d model output or 256 for the MLP model output.",
    )
    main(**vars(p.parse_args()))
