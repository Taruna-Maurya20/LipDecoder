# import tensorflow as tf
# from typing import List
# import cv2
# import os
# import numpy as np
# import imageio

# # Vocabulary definition
# vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
# char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")

# # # Mapping integers back to original characters
# # num_to_char = tf.keras.layers.StringLookup(
# #     vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
# # )
# # Define character mapping (1-based indexing)
# vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
# char_map = {i: c for i, c in enumerate(vocab, 1)}

# def num_to_char(sequence):
#     """Convert numeric token output to readable text."""
#     return ''.join([char_map[c] for c in sequence if c > 0 and c in char_map])

# # Function to load video frames
# def load_video(path:str) -> List[np.ndarray]: 
#     cap = cv2.VideoCapture(path)
    
#     if not cap.isOpened():
#         print(f"Error: Cannot open video file {path}")
#         return None

#     frames = []
#     for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Error: Unable to read frame from {path}")
#             break
        
#         # Convert to grayscale and crop
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame = frame[190:236,80:220]

#         # Normalize and convert to uint8 (0-255)
#         frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
        
#         # Convert to 3-channel grayscale image (needed for GIF)
#         frame = np.stack([frame] * 3, axis=-1)
#         frames.append(frame)

#     cap.release()

#     if not frames:
#         print(f"Error: No frames read from {path}")
#         return None

#     return frames
    
# # Function to load alignments from .align file
# def load_alignments(path:str) -> List[str]: 
#     if not os.path.exists(path):
#         print(f"Error: Alignment file not found: {path}")
#         return None

#     with open(path, 'r') as f: 
#         lines = f.readlines() 

#     tokens = []
#     for line in lines:
#         line = line.split()
#         if len(line) >= 3 and line[2] != 'sil': 
#             tokens = [*tokens, ' ', line[2]]

#     if not tokens:
#         print(f"Error: No valid tokens found in alignment file {path}")
#         return None

#     return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

# # Function to load video and corresponding alignments
# def load_data(path: str): 
#     path = bytes.decode(path.numpy())
#     file_name = path.split('/')[-1].split('.')[0]
#     file_name = path.split('\\')[-1].split('.')[0]

#     base_dir = 'C:\\Users\\Abhilasha\\LipNet\\data'
#     video_path = os.path.join(base_dir, 's1', f'{file_name}.mpg')
#     alignment_path = os.path.join(base_dir, 'alignments', 's1', f'{file_name}.align')

#     print(f"Video path: {video_path}")
#     print(f"Alignment path: {alignment_path}")

#     if not os.path.exists(video_path):
#         print(f"Error: Video file not found: {video_path}")
#         return None, None

#     frames = load_video(video_path)
#     if frames is None:
#         print(f"Error: Failed to load video data from {video_path}")
#         return None, None

#     if not os.path.exists(alignment_path):
#         print(f"Error: Alignment file not found: {alignment_path}")
#         return frames, None

#     alignments = load_alignments(alignment_path)
#     if alignments is None:
#         print(f"Error: Failed to load alignment data from {alignment_path}")
#         return frames, None

#     return frames, alignments

# # Main script example
# if __name__ == "__main__":
#     file_path = tf.convert_to_tensor("C:\\Users\\Abhilasha\\LipNet\\data\\s1\\sample.mpg")
#     video, annotations = load_data(file_path)

#     if video is None:
#         print("Video loading failed.")
#     if annotations is None:
#         print("Alignment loading failed or missing.")
    
#     # Save video as GIF
#     if video:
#         imageio.mimsave('animation.gif', video, fps=10)
#         print("GIF saved as 'animation.gif'")



# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# import imageio
# import string

# # Vocabulary definition
# vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
# char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")

# # Mapping integers back to original characters
# num_to_char = tf.keras.layers.StringLookup(
#     vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
# )

# # Define character mapping for manual conversion
# char_map = {i: c for i, c in enumerate("abcdefghijklmnopqrstuvwxyz' ", 1)}

# def manual_num_to_char(sequence):
#     """Convert numeric token output to readable text using a dictionary."""
#     return ''.join([char_map[c] for c in sequence if c > 0 and c in char_map])

# # Function to load video frames
# def load_video(path: str):
#     cap = cv2.VideoCapture(path)
    
#     if not cap.isOpened():
#         print(f"Error: Cannot open video file {path}")
#         return None

#     frames = []
#     for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Error: Unable to read frame from {path}")
#             break

#         # Convert to grayscale and crop
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame = frame[190:236, 80:220]

#         # Normalize and convert to uint8 (0-255)
#         frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)

#         # Convert to 3-channel grayscale image (needed for GIF)
#         frame = np.stack([frame] * 3, axis=-1)
#         frames.append(frame)

#     cap.release()

#     if not frames:
#         print(f"Error: No frames read from {path}")
#         return None

#     return frames

# # Function to load alignments from .align file
# def load_alignments(path: str):
#     if not os.path.exists(path):
#         print(f"Error: Alignment file not found: {path}")
#         return None

#     with open(path, 'r') as f:
#         lines = f.readlines()

#     tokens = []
#     for line in lines:
#         line = line.split()
#         if len(line) >= 3 and line[2] != 'sil':
#             tokens = [*tokens, ' ', line[2]]

#     if not tokens:
#         print(f"Error: No valid tokens found in alignment file {path}")
#         return None

#     return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

# # Function to load video and corresponding alignments
# def load_data(path: str):
#     path = bytes.decode(path.numpy())
#     file_name = os.path.splitext(os.path.basename(path))[0]

#     base_dir = 'C:\\Users\\Abhilasha\\LipNet\\data'
#     video_path = os.path.join(base_dir, 's1', f'{file_name}.mpg')
#     alignment_path = os.path.join(base_dir, 'alignments', 's1', f'{file_name}.align')

#     print(f"Video path: {video_path}")
#     print(f"Alignment path: {alignment_path}")

#     if not os.path.exists(video_path):
#         print(f"Error: Video file not found: {video_path}")
#         return None, None

#     frames = load_video(video_path)
#     if frames is None:
#         print(f"Error: Failed to load video data from {video_path}")
#         return None, None

#     if not os.path.exists(alignment_path):
#         print(f"Error: Alignment file not found: {alignment_path}")
#         return frames, None

#     alignments = load_alignments(alignment_path)
#     if alignments is None:
#         print(f"Error: Failed to load alignment data from {alignment_path}")
#         return frames, None

#     return frames, alignments

# # Function to predict and decode output
# def predict(video, model):
#     """Predicts the lip movements and decodes them into text."""
#     video_expanded = np.expand_dims(video, axis=-1)  # Add channel dimension
#     video_expanded = np.expand_dims(video_expanded, axis=0)  # Add batch dimension

#     # Get model predictions
#     yhat = model.predict(video_expanded)

#     # Decode output tokens
#     predicted_tokens = tf.keras.backend.ctc_decode(yhat, [yhat.shape[1]], greedy=True)[0][0]

#     # Convert tensor to a list and remove -1 values
#     decoded_text = manual_num_to_char(predicted_tokens.numpy().tolist())

#     print(f"Decoded Output: {decoded_text}")
#     return decoded_text





import os
import cv2
import numpy as np
import tensorflow as tf
import imageio

# Vocabulary definition (Ensuring unique entries)
ALL_LABELS = [
    *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ "),  # Uppercase letters and space
    *['CH', 'SH', 'TH', 'EE', 'OO', 'AY', 'AR', 'ER', 'OW', 'UH', 'IH', 'B', 'D']  # Phonemes
]

# Ensure the vocabulary has unique values (remove duplicates)
ALL_LABELS = list(set(ALL_LABELS))  # Removes duplicates
ALL_LABELS.sort()  # Sort the list to ensure consistent ordering

# Create mapping from word labels (e.g., SIL, BIN, BLUE...) to indices
char_to_num = tf.keras.layers.StringLookup(vocabulary=ALL_LABELS, oov_token="", mask_token="")

# Create reverse mapping (from indices back to word labels)
num_to_char = tf.keras.layers.StringLookup(vocabulary=ALL_LABELS, invert=True, oov_token="", mask_token="")

# Manually decode numeric sequence to characters using num_to_char
def manual_num_to_char(sequence):
    """Convert numeric token output to readable text using num_to_char."""
    return ''.join([num_to_char(c).numpy().decode('utf-8') for c in sequence if c > 0])

# Load video frames
def load_video(path: str):
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {path}")
        return None

    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to read frame from {path}")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[190:236, 80:220]

        frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
        frame = np.stack([frame] * 3, axis=-1)
        frames.append(frame)

    cap.release()

    if not frames:
        print(f"Error: No frames read from {path}")
        return None

    return frames

# Load alignments (updated version)
def load_alignments(path: str):
    """Load words from the .align file (ignore start and end times)."""
    if not os.path.exists(path):
        print(f"❌ Error: Alignment file not found: {path}")
        return None

    words = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                _, _, word = parts  # Ignore start, end times
                words.append(word.lower())  # Make lowercase for consistency

    # Return the list of words from the alignment file
    return words

# ✅ FINAL FIXED: Load video + alignment together
def load_data(path: str):
    path = bytes.decode(path.numpy())
    file_name = os.path.splitext(os.path.basename(path))[0]

    base_dir = os.path.join('..', 'data')
    video_path = os.path.join(base_dir, 's1', f'{file_name}.mpg')
    alignment_path = os.path.join(base_dir, 'alignments', 's1', f'{file_name}.align')

    print(f"Video path: {video_path}")
    print(f"Alignment path: {alignment_path}")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None, None

    frames = load_video(video_path)
    if frames is None:
        print(f"Error: Failed to load video data from {video_path}")
        return None, None

    if not os.path.exists(alignment_path):
        print(f"Error: Alignment file not found: {alignment_path}")
        return frames, None

    alignments = load_alignments(alignment_path)
    if alignments is None:
        print(f"Error: Failed to load alignment data from {alignment_path}")
        return frames, None

    return frames, alignments

