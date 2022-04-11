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
from scipy.io import loadmat
import cv2          # OpenCV : computer vision / image processing library
import numpy as np  # modules for basic tensors, scientific computation
import torch        # PyTorch, module for NN, models, passforward....
from sklearn.metrics import pairwise_distances  # module to compute distances
from tqdm import tqdm   # module to display loading bar
import sys
import cv2
import time
import browser

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
    bsldict_features_path,
    input_path: Path,
    keyword: str,
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
    msg = "Please check the BSLDict metadata"
    assert bsldict_metadata_path.exists(), msg

    print(f"Loading BSLDict metadata from {bsldict_metadata_path}")

    with open(bsldict_metadata_path, "rb") as f:    #"rb" : open a file with reading (r) in binary (b)
        bsldict_metadata = pkl.load(f)  #load pkl : fichier binaire --> obj python
    dict_ix = np.where(np.array(bsldict_metadata["videos"]["word"]) == keyword)[0]
    dict_words = np.array(bsldict_metadata["videos"]["word"]) # dict words
    dict_video_urls = np.array(bsldict_metadata["videos"]["video_link_db"]) # URLS of corresponding videos
    yt_ids = np.array(bsldict_metadata["videos"]["youtube_identifier_db"])
    del bsldict_metadata

    # check if the features are downloaded
    msg = "Please check the BSLDict featrues"
    assert bsldict_features_path.exists(), msg

    print(f"Loading BSLDict features from {bsldict_features_path}")

    mat = loadmat(bsldict_features_path)
    # out of the model for dict videos
    dict_features = [list(elt) for elt in mat["vid_features"]]
    del mat

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

    del rgb_orig

    # Sampling rgb : rgb_samples (sampling of video) and t_mid (corresponding (middle) timestamp)
    # It produce a tensor of num_clips of clips (each of num_in_frames frame) in RGB with rgb_input[i,j] shapes
    rgb_samples, t_mid = frame_sampler(rgb=rgb_input, num_in_frames=num_in_frames)
    #rgb_samples, t_mid = sliding_windows(rgb=rgb_input, num_in_frames=num_in_frames, stride=2)

    del rgb_input

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
    #avg_sim = np.mean(sim, axis=0)  # taking the average of all clips
    avg_sim = np.max(sim, axis=0)   # taking the proba max of all clips

    # sort the array and get indexes of the corresponding videos
    version_sorted_ix = np.flip(np.argsort(avg_sim, kind='quicksort'))


    if keyword != "NO_KEYWORD_TO_SPOT":
        for elt in dict_ix:
            print(dict_words[elt], " -- rank :", np.where(version_sorted_ix == elt)[0][0], " -- p = ", avg_sim[elt])
    else:
        for i in range(num_top):
             print(dict_words[version_sorted_ix[i]], " -- ", avg_sim[version_sorted_ix[i]], " -- ",
                   dict_video_urls[version_sorted_ix[i]])



    print("Saving html file")
    generate_html(filename="result.html",
    web_dir = "../",
    video_word = dict_words,
    video_prob = avg_sim,
    order_sorted = version_sorted_ix,
    video_url= dict_video_urls,
    yt_ids = yt_ids,
    num_videos_to_show=num_top)


def video_record(file_name:str):
    cap = cv2.VideoCapture(0)
    FOURCC = "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)

    assert cap.isOpened() == True, "Cannot connect to the camera"

    w = int(cap.get(3))
    h = int(cap.get(4))

    flag_record = False

    output = cv2.VideoWriter(file_name, fourcc, 25, (h, h))

    while (True):
        ret, frame = cap.read()
        cropped_frame = frame[:, w // 2 - h // 2:w // 2 + h // 2]

        if flag_record or cv2.waitKey(1) & 0xFF == ord(' '):
            flag_record = True

            output.write(cropped_frame)

            cv2.circle(cropped_frame, (h - h // 12, h // 12), 10, (0, 0, 254), -1)
            cv2.putText(cropped_frame, "REC", (h - h // 12 - 80, h // 12 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 254), 2)
        else:
            cv2.circle(cropped_frame, (h - h // 12, h // 12), 10, (0, 254, 0), -1)
            cv2.putText(cropped_frame, "STDBY", (h - h // 12 - 120, h // 12 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 254, 0), 2)

        cv2.imshow('Frame', cropped_frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()


def generate_html(
    filename,
    web_dir,
    video_word,
    video_prob,
    order_sorted,
    video_url,
    yt_ids,
    num_videos_to_show=20,
):
    """
    Generate an HTML page with the results

    """
    print(f"Saving to {web_dir}/{filename}")
    webpage = browser.HTMLBrowser(
        title=filename,
        refresh=True,
        filename=filename,
        web_dir=web_dir,
        header_template_path=None,
    )

    webpage.add_title("your results")
    a = "reverse sign langage test - sacha et ines"
    b = []
    for k in range(10):
        b.append(a)
    webpage.add_text_to_new_section(b)

    keys = ["qdsfsqdfsqdf"]*10
    links = ["http://youtube.com/"]*10
    probs = [0.7]*10
    refs = [0,1,0,1,0,1,0,1,0,1]
    thresholds = [0.1*k for k in range(50)]

    webpage.add_stats(keys, links, probs, refs, thresholds)



    vids = []
    txts = []
    links = []
    youtube_ids = []
    cnt = 0
    for i in range(num_videos_to_show):
            display_text = f" Prediction n°: {i+1} ;  {video_word[order_sorted[i]]} ; p = ({video_prob[order_sorted[i]]:.3f})"
            vids.append(video_url[order_sorted[i]])
            txts.append(display_text)
            links.append(video_url[order_sorted[i]])
            youtube_ids.append(yt_ids[order_sorted][i])

    webpage.add_videos(vids, txts, links, yt_ids = youtube_ids, width=250, cols_per_row=4, loop=0)

    webpage.save()


if __name__ == "__main__":

    file_name = "inputs/input-" + time.strftime("%Y%m%d-%H%M%S" + ".mp4")

    p = argparse.ArgumentParser(description="Helper script to run main.")
    p.add_argument(
        "--checkpoint_path",
        type=Path,
        default="../models/bsl1k_bobsl/i3d.pth.tar",
        help="Path to i3d model.",
    )
    p.add_argument(
        "--bsldict_metadata_path",
        type=Path,
        default="../bsldict/bsldict_v2_i3d.pkl",
        help="Path to bsldict data",
    )
    p.add_argument(
        "--bsldict_features_path",
        type=Path,
        default="../models/bsl1k_bobsl/bsldict_features.mat",
        help="Path to bsldict data",
    )
    p.add_argument(
        "--input_path",
        type=Path,
        default=file_name,
        help="Path to input video.",
    )
    p.add_argument(
        "--keyword",
        type=str,
        default="NO_KEYWORD_TO_SPOT",
        help="Spot a particular word",
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
        default=2281,
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


    if str(p.parse_args().input_path) == file_name:
        video_record(file_name)

    main(**vars(p.parse_args()))







