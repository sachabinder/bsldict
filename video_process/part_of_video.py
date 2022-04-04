import cv2

def extract_part(input_path: str, parts: dict):
    """
    Extract from an input video parts of this video, save them in .mp4.

    :param input_path: path of the video input
    :param parts: Dict {"name_of_the_part":(begin_frame, end_frame), ...}
    """

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(input_path)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print(f"Error opening {input_path}")

    ret, frame = cap.read() # read the first frame
    f = 0   # frame counter

    h, w, c = frame.shape   # get dims of video

    FOURCC = "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)

    # videos to save
    writers = [cv2.VideoWriter(file_name, fourcc, 25, (w, h)) for file_name in parts.keys()]
    print(f"Creating {len(writers)} clips...")


    while ret:
        for i, part in enumerate(parts.values()):
            start, end = part
            if start <= f <= end:
                writers[i].write(frame) # save good frames on corresponding videos

        # next frame
        ret, frame = cap.read()
        f += 1

    # close and saving writers
    for writer in writers:
        writer.release()

    # close capture
    cap.release()

    print("Done !")


def play_frame_by_frame(input_path: str):
    """
    Play a video frame by frame, with keys
        - O : previous frame
        - P : next frame
        - Q : quit

    :param input_path: path of the video input
    """
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(input_path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video  file")

    ret, frame = cap.read()  # read the first frame
    f = 0  # frame counter
    h, w, c = frame.shape  # get dims of video

    # Read until video is completed
    while ret:

        frame = cv2.rectangle(frame, (5, 5), (w//4, h//6), (255, 255, 255), thickness=-1)  # draw fill rectangle to put text into
        frame = cv2.putText(frame, str(f), (w//5 - 40, h//6 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        # Display the resulting frame
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('p'):
            # next frame
            ret, frame = cap.read()
            f += 1

        elif cv2.waitKey(25) & 0xFF == ord('o'):
            # next frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, f-1)
            ret, frame = cap.read()
            f -= 1

        # Press Q on keyboard to  exit
        elif cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    file = "apple.mp4"  # input file path
    parts = {"input_apple_test.mp4": (0, 69)}  # part of videos to extract with file name "file_name":(begin, end)

    #play_frame_by_frame(file)

    extract_part(file, parts)