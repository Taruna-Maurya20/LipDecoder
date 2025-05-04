# import os
# import cv2  # Make sure you have OpenCV installed

# def load_video(video_path):
#     """Load video frames from the specified video file."""
#     frames = []
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         print(f"Error: Could not open video {video_path}")
#         return frames

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)

#     cap.release()
#     return frames

# def load_alignment_data(alignment_path):
#     """Load alignment data from the specified file."""
#     alignments = []
#     with open(alignment_path, 'r') as file:
#         for line in file:
#             alignments.append(line.strip())  # Add your parsing logic here
#     return alignments

# def preprocess_data(video_directory, alignment_directory):
#     video_data = []
#     alignment_data = []

#     # List all video files in the video directory
#     video_files = os.listdir(video_directory)
#     print(f"Looking in: {video_directory}")
#     print("Video files found:", video_files)

#     for filename in video_files:
#         if filename.endswith('.mpg') or filename.endswith('.mp4') or filename.endswith('.avi'):
#             video_path = os.path.join(video_directory, filename)
#             frames = load_video(video_path)
#             video_data.append(frames)

#             # Generate alignment file path
#             alignment_file = filename.rsplit('.', 1)[0] + '.align'
#             alignment_path = os.path.join(alignment_directory, alignment_file)

#             if os.path.exists(alignment_path):
#                 alignments = load_alignment_data(alignment_path)
#                 alignment_data.append(alignments)
#             else:
#                 print(f"Alignment file not found for {filename}: {alignment_path}")

#     return video_data, alignment_data

# if __name__ == "__main__":
#     video_directory = 'C:/Users/Abhilasha/Desktop/lipnet2/data/s1'  # Path to video files
#     alignment_directory = 'C:/Users/Abhilasha/Desktop/lipnet2/data/alignments/s1/align'  # Path to alignment files
#     video_data, alignment_data = preprocess_data(video_directory, alignment_directory)

#     print(f"Total videos processed: {len(video_data)}")
#     print(f"Total alignments found: {len(alignment_data)}")

#     for i in range(len(video_data)):
#         if i < len(alignment_data):
#             print(f"Video {i + 1} alignments:", alignment_data[i])
#         else:
#             print(f"Video {i + 1} has no alignment data.")

import os
import cv2
import numpy as np

# Constants for resizing
IMG_HEIGHT = 46
IMG_WIDTH = 140
MAX_FRAMES = 75
CHAR_TO_INDEX = {
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6,
    'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13,
    'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20,
    'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, ' ': 26, "'": 27,
    '.': 28, ',': 29, '?': 30, '!': 31, '-': 32, ':': 33, ';': 34,
    '"': 35, '0': 36, '1': 37, '2': 38, '3': 39, '4': 40
}

def preprocess_video(video_path):
    """Reads a video, resizes and normalizes it to shape (MAX_FRAMES, 46, 140, 1)."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
        frames.append(resized)

    cap.release()

    # Pad if fewer than MAX_FRAMES
    while len(frames) < MAX_FRAMES:
        frames.append(np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8))

    video_np = np.array(frames, dtype=np.float32) / 255.0
    video_np = np.expand_dims(video_np, axis=-1)  # Add channel dim
    return video_np

def align_to_char_sequence(align_path):
    """Parses alignment file and returns list of character indices."""
    with open(align_path, 'r') as f:
        lines = f.readlines()
    
    chars = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            word = parts[2].lower()
            for char in word:
                if char in CHAR_TO_INDEX:
                    chars.append(CHAR_TO_INDEX[char])
                else:
                    chars.append(40)  # Default to blank for unknown chars
            chars.append(CHAR_TO_INDEX[' '])  # Add space after each word
    
    return chars

def load_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def load_alignment_data(alignment_path):
    alignments = []
    with open(alignment_path, 'r') as file:
        for line in file:
            alignments.append(line.strip())
    return alignments

def preprocess_data(video_directory, alignment_directory):
    video_data = []
    alignment_data = []
    video_files = os.listdir(video_directory)

    for filename in video_files:
        if filename.endswith(('.mpg', '.mp4', '.avi')):
            video_path = os.path.join(video_directory, filename)
            frames = load_video(video_path)
            video_data.append(frames)

            alignment_file = filename.rsplit('.', 1)[0] + '.align'
            alignment_path = os.path.join(alignment_directory, alignment_file)

            if os.path.exists(alignment_path):
                alignments = load_alignment_data(alignment_path)
                alignment_data.append(alignments)
            else:
                print(f"Alignment file not found for {filename}: {alignment_path}")

    return video_data, alignment_data
