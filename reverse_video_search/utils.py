 #=================== Importation of modules ===================
import math
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from matplotlib import gridspec
from PIL import Image
from tqdm import tqdm

sys.path.append("..")
#=================== Importation of dowload framework (wget and yt-dl) ===================
from bsldict.download_videos import download_hosted_video, download_youtube_video

#=================== Importation of the model framework ===================
from models.i3d import InceptionI3d


#=================== Visualisation processing ===================
def viz_similarities(
    rgb: torch.Tensor,
    t_mid: np.ndarray,
    sim: np.ndarray,
    similarity_thres: float,
    keyword: str,
    output_path: Path,
    viz_with_dict: bool,
    dict_video_links: tuple,
    num_in_frames: int
):
    """
    Save a visualization video for similarities
        between the input video and dictionary videos.
    1) Create a figure for every frame of the input video.
    2) On the left: show the input video frame, with the search keyword below
        (the keyword is displayed as red if below the maximum similarity is below
        similarity threshold, as green otherwise).
    3) On the right: show the plots, each corresponding to a different dictionary version,
        display the maximum similarity for the given frame.
    4) At the top: show side-by-side the middle frames of the dictionary videos
        corresponding to the keyword.

    :param rgb: input video
    :param t_mid: ....?
    :param sim: array of similarities
    :param similarity_thres: threshold of similarity
    :param keyword: keyword to spot
    :param output_path: output_path of the final file
    :param viz_with_dict: True if you want BSLDict picture to be in the output video
    :param dict_video_links: array of dict_video_urls and dict_youtube_ids
    :param num_in_frames: number of frames processed at a time by the model
    """
    # =================== formatting the keyword ===================
    # Put linebreaks for long strings every 40 chars
    keyword = list(keyword)
    max_num_chars_per_line = 40
    num_linebreaks = int(len(keyword) / max_num_chars_per_line)
    for lb in range(num_linebreaks):
        pos = (lb + 1) * max_num_chars_per_line
        keyword.insert(pos, "\n")
    keyword = "".join(keyword)
    keyword = f"Keyword: {keyword}"

    num_frames = rgb.shape[1]   # number of frames
    height = rgb.shape[2]   # height of the input
    offset = height / 14    # horizontal position of the keyword

    fig = plt.figure(figsize=(9, 3))    # creating the figure
    # 900, 300
    # width and height of the figure
    figw, figh = fig.get_size_inches() * fig.dpi
    figw, figh = int(figw), int(figh)

    # creating subplots
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2]) # grid specification 1 row, 2 colls
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # similarity plot
    sim_plots = ax2.plot(range(int(num_in_frames / 2), int(num_in_frames / 2) + sim.shape[0]), sim) # num_in_frame/2 to center the plot (because model is not fed img by img)
    ax2.set_ylabel("Similarity")
    ax2.set_xlabel("Time")
    ax2.set_xlim(0, num_frames - 1)
    # Legend of sim plot (name of videos from BSLDict)
    num_versions = sim.shape[1]
    plt.legend([f"v{v + 1}" for v in range(num_versions)], loc="upper right")

    # Show the middle frame of each version used from dict
    if viz_with_dict:
        res = 256   # resolution of images to show
        dict_video_urls, dict_youtube_ids = dict_video_links    # extract url form wget and ids for yt-download
        num_dicts = len(dict_video_urls)    # number of version of keyword extracted from the dict
        stacked_dicts = np.zeros((num_dicts, res, res, 3))  # tensor of photos to show

        # get corresponding images process
        for v, dict_vid_url in enumerate(dict_video_urls):
            dict_color = sim_plots[v].get_color()   # get the color of the plot
            # dict_color = list(mcolors.TABLEAU_COLORS.values())[0]
            yid = dict_youtube_ids[v]   # YT id of the video if on YT, else None

            # download and store the frame of corresponding video v
            dict_frame = get_dictionary_frame(
                dict_vid_url, yid, v=f"v{v + 1}", color=dict_color, res=res
            )
            stacked_dicts[v] = dict_frame   # store frame in the tensor


        dict_viz = np.hstack(stacked_dicts) # transform the collection of 5 img [256x256] on a single img [256, 5*256] (side-by-side images)
        dh, dw, _ = dict_viz.shape
        dh, dw = int(figw * dh / dw), figw  # keep the ratio dh/dw
        dict_viz = cv2.resize(dict_viz, (dw, dh)) # resizing to be put in figure
    else:
        dh = 0

    # Create videowriter
    print(f"Saving visualization to {output_path}")
    FOURCC = "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    out_video = cv2.VideoWriter(str(output_path), fourcc, 25, (figw, figh + dh))

    # Generate the video output
    for t in tqdm(range(num_frames)):
        # input vizualisation
        img = cv2.resize(im_to_numpy(rgb[:, t]), (256, 256)) # resize each img of the input video
        ax1.imshow(img) # plot the img
        ax1.set_title("Continuous input")

        # Title of similarity plot
        t_ix = abs(t_mid - t).argmin() # index of the nearest clips (sliding window)
        sim_t = max(sim[t_ix, :]) # max similarity at time step t
        sim_t_ix = sim[t_ix, :].argmax() # video which is max sim at time step t
        dict_color = sim_plots[sim_t_ix].get_color()    # color of the video which is max sim
        sim_text = f"Max similarity: {sim_t:.2f}"
        ax2.set_title(sim_text, color=dict_color)   # put the title on the greate color

        # timeline animation scale of F (because the model is feed with F frames)
        time_line = ax2.axvline(x=t_ix)
        time_rect = ax2.add_patch(
            patches.Rectangle(
                (t_ix, ax2.get_ylim()[0]), num_in_frames, np.diff(ax2.get_ylim())[0], alpha=0.5,
            )
        )

        # highlight the detection of sign on the input video figure
        sim_color = "red"
        if sim_t >= similarity_thres:
            sim_color = "green"
            # Rectangle whenever above a sim threshold
            ax1.add_patch(
                patches.Rectangle(
                    (0, 0), 256, 256, linewidth=10, edgecolor="green", facecolor="none"
                )
            )

        # Display keyword
        ax1.text(
            offset,
            256,
            keyword,
            fontsize=12,
            fontweight="bold",
            color="white",
            verticalalignment="top",
            bbox=dict(facecolor=sim_color, alpha=0.9),
        )
        ax1.axis("off") # remove axis from the figure


        if t == 0:
            plt.tight_layout()
        fig_img = fig2data(fig)
        fig_img = np.array(Image.fromarray(fig_img))
        if viz_with_dict:
            fig_img = np.vstack((dict_viz, fig_img)) # join dict images and plt figure on a sigle frame

        # add the frame to the video
        out_video.write(fig_img[:, :, (2, 1, 0)].astype("uint8")) # put it from RGB to BGR (for oCV)

        # clear axis for the next step
        ax1.clear()
        time_line.remove()
        time_rect.remove()

    # finish and save the video
    out_video.release()

    # check if the output path exist
    msg = (f"Did not find a generated video at {output_path}, is the FOURCC {FOURCC} "
           f"supported by your opencv install?")
    assert output_path.exists(), msg



def fig2data(fig: plt.figure) -> np.ndarray:
    """
    Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it

    :param fig: a matplotlib figure
    """
    # draw the renderer
    fig.canvas.draw()

    # get width and height of the figure
    w, h = fig.canvas.get_width_height()
    # Get the RGBA buffer from the figure
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return buf



# =================== acquisition processing ===================

def get_dictionary_frame(url: str, yid: str, v: str, color: str, res: int =256) -> np.ndarray:
    """
    1) Download the dictionary video to a temporary file location
        (with youtube-dl if it has a youtube-identifier, otherwise with wget),
        if cannot download, return black image,
    2) Read the middle frame of the video,
    3) Resize to a square resolution determined by `res`,
    4) Put a rectangle around the frame with `color` (convert matplotlib color to cv2 color),
    5) Put a text to display the dictionary version number (determined by `v`)
    6) Convert BGR to RGB and return the frame
    7) Remove the temporary download

    :param url: video url
    :param yid: youtube_identifier (can be None)
    :param v: string to display on top of the frame, denotes the dictionary version number
    :param color: color in matplotlib style, to be used in the rectangle and text
    :param res: resolution of the frame at which to resize
    """
    try:
        # Temporary file location, importation
        tmp = f"tmp_{time.time()}.mp4"
        if yid:
            # Download with youtube-dl if on YT
            download_youtube_video(yid, tmp)
        else:
            # Download with wget else
            download_hosted_video(url, tmp)

        # Read the video
        cap = cv2.VideoCapture(tmp)
        # Get the total number of frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Read the middle frame
        cap.set(propId=1, value=math.floor(frame_count / 2))
        ret, frame = cap.read()

        # work on the middle frame
        frame = cv2.resize(frame, (res, res))   # resizing

        # Version color (matplotlib -> cv2)
        rgb_color = mcolors.to_rgb(color)   # plot color (for the colored square)
        bgr_color = tuple((255 * rgb_color[2], 255 * rgb_color[1], 255 * rgb_color[0])) # RGB in [0,1] -> BGR in [0,255]

        # draw the color rectangle on frame
        frame = cv2.rectangle(frame, (0, 0), (res - 1, res - 1), color=bgr_color,thickness=20)
        # Version text
        frame = cv2.rectangle(frame, (30, 20), (80, 60), color=bgr_color, thickness=-1) # draw fill rectangle to put text into
        frame = cv2.putText(
            frame, v, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4
        )

        # BGR -> RGB
        frame = frame[:, :, [2, 1, 0]]
        # Remove temporary download
        os.remove(tmp)
        return frame
    except:
        print(f"Could not download dictionary video {url}")
        return np.zeros((res, res, 3))


def load_rgb_video(video_path: Path, fps: int) -> torch.Tensor:
    """
    Load frames of a video using cv2  and normalized it [0,255] -> [0,1]
    (fetch from provided URL if file is not found at given location).

    :param video_path: path of the video to load
    :param fps: frame rate of the video
    """

    # capture of the video and all params
    cap = cv2.VideoCapture(str(video_path))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    # cv2 won't be able to change frame rates for all encodings, so we use ffmpeg
    if cap_fps != fps:
        # changing the frame rate of the video with ffmpeg
        tmp_video_path = f"{video_path}.tmp.{video_path.suffix}"
        shutil.move(video_path, tmp_video_path)
        cmd = (
            f"ffmpeg -i {tmp_video_path} -pix_fmt yuv420p "
            f"-filter:v fps=fps={fps} {video_path}"
        )
        print(f"Generating new copy of video with frame rate {fps}")
        os.system(cmd)
        Path(tmp_video_path).unlink()

        # recapture of the video and all parameters
        cap = cv2.VideoCapture(str(video_path))
        cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_fps = cap.get(cv2.CAP_PROP_FPS)

        # check if all is good
        assert cap_fps == fps, f"ffmpeg failed to produce a video at {fps}"

    # convert everything in a torch.Tensor
    f = 0   # frame counter
    rgb = []
    while True:
        # frame: BGR, (h, w, 3), dtype=uint8 0..255
        ret, frame = cap.read() # read frame by frame
        if not ret:
            break
        # BGR (OpenCV) -> RGB (Torch)
        frame = frame[:, :, [2, 1, 0]]
        rgb_t = im_to_torch(frame)  # casting the frame in torch.Tensor and normalize it
        rgb.append(rgb_t)
        f += 1
    cap.release()
    # (nframes, RGB, cap_height, cap_width) -> (RGB, nframes, cap_height, cap_width)
    rgb = torch.stack(rgb).permute(1, 0, 2, 3)
    print(
        f"Loaded video {video_path} with {f} frames [{cap_height}hx{cap_width}w] res. ",
        f"at {cap_fps}"
    )
    return rgb



#=================== Images/Tensors processing ===================

def prepare_input(
    rgb: torch.Tensor,
    resize_res: int = 256,
    inp_res: int = 224,
    mean: torch.Tensor = 0.5 * torch.ones(3),
    std: torch.Tensor = 1.0 * torch.ones(3),
) -> torch.Tensor:
    """
    Process the video:
    1) Resize to [resize_res x resize_res]
    2) Center crop with [inp_res x inp_res]
    3) Color normalize using mean/std

    :param rgb: input video tensor
    :param resize_res: resolution of resizing
    :param inp_res: input resolution need to feed the model
    :param mean: tensor that allows to get the color mean of a frame (for normalization)
    :param std: standard tensor (for normalization)
    """
    # videos characteristics : iC (number of colors), iF (number of frames), iH (height), iW (width)
    iC, iF, iH, iW = rgb.shape

    # create an empty np tensor that represents the video with output resolutions
    rgb_resized = np.zeros((iF, resize_res, resize_res, iC))
    # fill in this tensor
    for t in range(iF):
        tmp = rgb[:, t, :, :] # extract the image t from the tensor
        rgb_resized[t] = cv2.resize(im_to_numpy(tmp), (resize_res, resize_res)) # resizing

    # reorder indices (numFrames, height, widht, BGR) -> (BGR, numFrames, height, width)
    rgb = np.transpose(rgb_resized, (3, 0, 1, 2))

    # Center crop coords
    ulx = int((resize_res - inp_res) / 2)
    uly = int((resize_res - inp_res) / 2)

    # Crop 256x256
    rgb = rgb[:, :, uly : uly + inp_res, ulx : ulx + inp_res]
    rgb = to_torch(rgb).float()
    assert rgb.max() <= 1

    # normalization
    rgb = color_normalize(rgb, mean, std)
    return rgb



def sliding_windows(rgb: torch.Tensor, num_in_frames: int, stride: int,) -> tuple:
    """
    Return sliding windows and corresponding (middle) timestamp

    :param rgb: RGB video
    :param num_in_frames: number of frames processed at a time by the model
    :param stride: how many frames to stride
    """
    # videos characteristics : C (number of colors), nFrames (number of frames), H (height), W (width)
    C, nFrames, H, W = rgb.shape

    # If needed, pad to the minimum clip length (because the model will not be able to work otherwise)
    if nFrames < num_in_frames:
        rgb_ = torch.zeros(C, num_in_frames, H, W)
        rgb_[:, :nFrames] = rgb
        rgb_[:, nFrames:] = rgb[:, -1].unsqueeze(1)
        rgb = rgb_
        nFrames = rgb.shape[1]

    num_clips = math.ceil((nFrames - num_in_frames) / stride) + 1 # number of clips
    assert num_clips > 0, f"Your input video of {nFrames} frame(s) cannot be slided with {num_in_frames} frame(s) at " \
                           f"the same time in the model."


    plural = "" # jsut for printing
    # if there are enough images to have multiple clips
    if num_clips > 1:
        plural = "s"
    print(f"{num_clips} clip{plural} resulted from sliding window processing.")

    # declaration of outputs
    rgb_slided = torch.zeros(num_clips, 3, num_in_frames, H, W)
    t_mid = []

    # For each clip
    for j in range(num_clips):
        # Check if num_clips becomes 0
        actual_clip_length = min(num_in_frames, nFrames - j * stride)

        # compute beginning time
        if actual_clip_length == num_in_frames:
            t_beg = j * stride  # if the t_beg + stride <= nFrames, no problem
        else:
            t_beg = nFrames - num_in_frames # we loop on the end of video
        t_mid.append(t_beg + num_in_frames / 2) # append middle time of each clips

        rgb_slided[j] = rgb[:, t_beg : t_beg + num_in_frames, :, :]  # append slided windows

    return rgb_slided, np.array(t_mid)


def im_to_numpy(img: torch.Tensor) -> np.ndarray:
    """
    Convert a torch.Tensor image on a numpy array image.

    :param img: tensor of an image with format (Color, Height, Widht)
    """
    img = to_numpy(img) # convert a torch Tensor to a numpy one
    img = np.transpose(img, (1, 2, 0))  # torch (Color, Height, Width) -> numpy (Height, Width, Color)
    return img


def im_to_torch(img: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy array image on a torch.Tensor image and normalize colors [0,255] -> [0,1].

    :param img: array of an image with format (Height, Widht, Color)
    """
    img = np.transpose(img, (2, 0, 1))  # numpy (Height, Width, Color) -> torch (Color, Height, Width)
    img = to_torch(img).float() # convert a numpy Tensor to a torch one
    if img.max() > 1:
        img /= 255
    return img


def to_numpy(tensor: torch.Tensor) -> numpy.ndarray:
    """
    Convert a torch Tensor on a Numpy array

    :param tensor: the torch tensor to convert
    """
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy() # move the tensor on the cpu and convert it

    elif type(tensor).__module__ != "numpy":
        raise ValueError(f"Cannot convert {type(tensor)} to numpy array")

    return tensor


def to_torch(ndarray: np.ndarray) -> torch.Tensor:
    """
    Convert a Numpy array on a torch Tensor.

    :param ndarray: the array to convert
    """
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError(f"Cannot convert {type(ndarray)} to torch tensor")
    return ndarray


def color_normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor of images by subtracting the mean, dividing by std.

    :param x: tensor of images to normalize
    :param mean: tensor that allows to get the color mean of a frame (for normalization)
    :param std: standard tensor (for normalization)
    """
    if x.dim() in {3, 4}:
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        assert x.size(0) == 3, "For single video format, expected RGB along first dim"
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
    elif x.dim() == 5:
        assert (
            x.shape[1] == 3
        ), "For batched video format, expected RGB along second dim"
        x[:, 0].sub_(mean[0]).div_(std[0])
        x[:, 1].sub_(mean[1]).div_(std[1])
        x[:, 2].sub_(mean[2]).div_(std[2])
    return x



#=================== Model loading ===================

def load_model(checkpoint_path: Path) -> torch.nn.Module:
    """
    Load pre-trained checkpoint, put in eval mode.

    :param checkpoint_path: path of i3d_mlp model
    :param return_i3d_embds: whether to return the intermediate i3d embeddings from the i3d_mlp model
    """
    model = InceptionI3d(num_classes=2281, num_in_frames=16, include_embds=True)
    checkpoint = torch.load(str(checkpoint_path))
    model.load_state_dict(checkpoint)
    # model = torch.nn.DataParallel(model)  # .cuda()
    model.eval() # for evaluation safety
    return model
