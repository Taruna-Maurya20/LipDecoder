# import numpy as np
# import cv2
# import os
# from train_and_save_model import load_model

# # === Constants ===
# VIDEO_PATH = r"C:\Users\Abhilasha\Desktop\lipbuddy\data\s1\bbaf2n.mpg"
# ALIGN_PATH = r"C:\Users\Abhilasha\Desktop\lipbuddy\data\alignments\s1\bbaf2n.align"
# FRAME_COUNT = 75
# IMG_HEIGHT = 46
# IMG_WIDTH = 140

# # === Label Map ===
# label_map = {
#     0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
#     5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
#     10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
#     15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
#     20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
#     25: 'Z', 26: ' ', 27: 'CH', 28: 'SH', 29: 'TH',
#     30: 'EE', 31: 'OO', 32: 'AY', 33: 'AR', 34: 'ER',
#     35: 'OW', 36: 'UH', 37: 'IH', 38: 'B', 39: 'D',
#     40: ''  # blank
# }

# # === Function: Read and preprocess video ===
# def load_and_preprocess_video(path):
#     cap = cv2.VideoCapture(path)
#     frames = []
#     while len(frames) < FRAME_COUNT:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
#         frames.append(resized)
#     cap.release()

#     # Pad or trim to 75 frames
#     if len(frames) < FRAME_COUNT:
#         frames += [np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)] * (FRAME_COUNT - len(frames))
#     elif len(frames) > FRAME_COUNT:
#         frames = frames[:FRAME_COUNT]

#     video_array = np.array(frames)
#     video_array = video_array[..., np.newaxis]  # (75, 46, 140, 1)
#     video_array = video_array[np.newaxis, ...]  # (1, 75, 46, 140, 1)
#     return video_array / 255.0

# # === Function: Read alignment file ===
# def read_align_file(filepath):
#     with open(filepath, 'r') as f:
#         words = [line.strip().split()[-1] for line in f.readlines()]
#     return ' '.join(words)

# # === Function: Decode model output ===
# def decode_predictions(predictions):
#     predicted_indices = np.argmax(predictions[0], axis=-1)

#     def collapse_repeats(indices):
#         result = []
#         prev = -1
#         for idx in indices:
#             if idx != prev and idx != 40:
#                 result.append(idx)
#             prev = idx
#         return result

#     collapsed = collapse_repeats(predicted_indices)
#     predicted_chars = [label_map[i] for i in collapsed]
#     return ''.join(predicted_chars)

# # === Main ===
# if __name__ == "__main__":
#     print("📥 Loading model...")
#     model = load_model()
#     print("🎥 Loading video...")
#     video_input = load_and_preprocess_video(VIDEO_PATH)
#     print("📊 Predicting...")
#     predictions = model.predict(video_input)

#     predicted_text = decode_predictions(predictions)
#     print("🔤 Predicted Text:", predicted_text)

#     ground_truth = read_align_file(ALIGN_PATH)
#     print("📖 Ground Truth:", ground_truth)




# import numpy as np
# import cv2
# import tensorflow as tf
# from train_and_save_model import load_model

# # === Helper: Collapse Repeats and Remove Blanks ===
# def collapse_repeats(indices):
#     result = []
#     prev = -1
#     for idx in indices:
#         if idx != prev and idx != 40:  # 40 is blank
#             result.append(idx)
#         prev = idx
#     return result

# # === Label Map (index to character) ===
# label_map = {
#     0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
#     5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
#     10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
#     15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
#     20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
#     25: 'Z', 26: ' ', 27: 'CH', 28: 'SH', 29: 'TH',
#     30: 'EE', 31: 'OO', 32: 'AY', 33: 'AR', 34: 'ER',
#     35: 'OW', 36: 'UH', 37: 'IH', 38: 'B', 39: 'D',
#     40: ''  # blank
# }

# # === Load Video and Preprocess ===
# def load_video(video_path, target_frames=75, height=46, width=140):
#     cap = cv2.VideoCapture(video_path)
#     frames = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         resized = cv2.resize(gray, (width, height))
#         normalized = resized / 255.0
#         frames.append(normalized)

#     cap.release()

#     if len(frames) < target_frames:
#         while len(frames) < target_frames:
#             frames.append(frames[-1])
#     else:
#         frames = frames[:target_frames]

#     video_np = np.array(frames)
#     video_np = video_np.reshape(1, target_frames, height, width, 1)
#     return video_np

# # === Load Alignment ===
# def load_alignment(align_path):
#     with open(align_path, 'r') as f:
#         lines = f.readlines()
#     words = [line.strip().split()[-1] for line in lines if not line.strip().endswith('sil')]
#     return ' '.join(words)

# # === Main Execution ===
# video_path = r"C:\Users\Abhilasha\Desktop\lipbuddy\data\s1\bbaf2n.mpg"
# align_path = r"C:\Users\Abhilasha\Desktop\lipbuddy\data\alignments\s1\bbaf2n.align"

# print("\U0001F4E5 Loading model...")
# model = load_model()
# print("✅ Model weights loaded from models/complete_model.h5")

# print("🎥 Loading video...")
# video_input = load_video(video_path)

# print("📊 Predicting...")
# predictions = model.predict(video_input)

# predicted_indices = np.argmax(predictions[0], axis=-1)
# collapsed = collapse_repeats(predicted_indices)
# predicted_chars = [label_map[i] for i in collapsed]
# predicted_text = ''.join(predicted_chars)

# # Load ground truth
# true_text = load_alignment(align_path)

# print("\U0001F4A4 Predicted Text:", predicted_text)
# print("\U0001F4D6 Ground Truth:", true_text)




# import numpy as np
# import cv2
# import os
# from train_and_save_model import load_model

# # === Constants ===
# VIDEO_PATH = r"C:\Users\Abhilasha\Desktop\lipbuddy\data\s1\bbaf2n.mpg"
# ALIGN_PATH = r"C:\Users\Abhilasha\Desktop\lipbuddy\data\alignments\s1\bbaf2n.align"
# FRAME_COUNT = 75
# IMG_HEIGHT = 46
# IMG_WIDTH = 140

# # === Label Map (41 classes) ===
# label_map = {
#     0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
#     5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
#     10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
#     15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
#     20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
#     25: 'Z', 26: ' ', 27: 'CH', 28: 'SH', 29: 'TH',
#     30: 'EE', 31: 'OO', 32: 'AY', 33: 'AR', 34: 'ER',
#     35: 'OW', 36: 'UH', 37: 'IH', 38: 'B', 39: 'D',
#     40: ''  # blank for CTC
# }

# # === Video Loader ===
# def load_and_preprocess_video(path):
#     cap = cv2.VideoCapture(path)
#     frames = []
#     while len(frames) < FRAME_COUNT:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
#         frames.append(resized)
#     cap.release()

#     # Ensure 75 frames
#     while len(frames) < FRAME_COUNT:
#         frames.append(np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8))
#     if len(frames) > FRAME_COUNT:
#         frames = frames[:FRAME_COUNT]

#     video_array = np.array(frames)
#     video_array = video_array[..., np.newaxis]  # Add channel dim
#     video_array = video_array[np.newaxis, ...]  # Add batch dim
#     return video_array / 255.0

# # === Read ground truth from .align file ===
# def read_align_file(filepath):
#     words = []
#     with open(filepath, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 3:
#                 word = parts[2]
#                 if word not in ['sil', 'sp']:
#                     words.append(word)
#     return ' '.join(words)

# # === Decode model prediction using CTC logic ===
# def decode_predictions(predictions):
#     predicted_indices = np.argmax(predictions[0], axis=-1)

#     # Collapse repeated values and remove blank (40)
#     decoded = []
#     prev = -1
#     for idx in predicted_indices:
#         if idx != prev and idx != 40:
#             decoded.append(idx)
#         prev = idx

#     predicted_chars = [label_map[i] for i in decoded]
#     return ''.join(predicted_chars)

# # === MAIN ===
# if __name__ == "__main__":
#     print("📥 Loading model...")
#     model = load_model()

#     print("🎥 Loading and preprocessing video...")
#     video_input = load_and_preprocess_video(VIDEO_PATH)

#     print("📊 Running prediction...")
#     predictions = model.predict(video_input)

#     predicted_text = decode_predictions(predictions)
#     ground_truth = read_align_file(ALIGN_PATH)

#     print("\n🔤 Predicted Text:", predicted_text)
#     print("📖 Ground Truth  :", ground_truth)


import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPool3D, Activation, TimeDistributed, Flatten, Bidirectional, LSTM, Dropout, Dense

# === Constants ===
DATA_DIR = r"C:\Users\Abhilasha\Desktop\lipbuddy\data\s1"
ALIGN_DIR = r"C:\Users\Abhilasha\Desktop\lipbuddy\data\alignments\s1"
IMG_HEIGHT = 46
IMG_WIDTH = 140
MAX_FRAMES = 75
NUM_CLASSES = 41  # 0–40 including blank

# === Character Map ===
CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
PHONES = ['CH', 'SH', 'TH', 'EE', 'OO', 'AY', 'AR', 'ER', 'OW', 'UH', 'IH', 'B', 'D']
ALL_LABELS = CHARS + PHONES
CHAR_MAP = {ch: idx for idx, ch in enumerate(ALL_LABELS)}
CHAR_MAP[''] = 40  # blank class

# === Model ===
def create_model():
    model = Sequential()
    model.add(tf.keras.Input(shape=(MAX_FRAMES, IMG_HEIGHT, IMG_WIDTH, 1)))

    model.add(Conv3D(128, 3, padding='same', activation='relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same', activation='relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same', activation='relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
    return model

# === Preprocess Video ===
def preprocess_video(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
        frames.append(resized)
    cap.release()

    if len(frames) < MAX_FRAMES:
        frames += [np.zeros((IMG_HEIGHT, IMG_WIDTH))] * (MAX_FRAMES - len(frames))
    elif len(frames) > MAX_FRAMES:
        frames = frames[:MAX_FRAMES]

    video = np.array(frames)[..., np.newaxis] / 255.0
    return video

# === Read Alignment File ===
def align_to_char_sequence(filepath):
    with open(filepath, 'r') as f:
        words = [line.strip().split()[-1] for line in f.readlines()]
    sentence = ' '.join(words)
    return [CHAR_MAP.get(ch.upper(), 40) for ch in list(sentence) if ch.upper() in CHAR_MAP or ch == ' ']

# === Load Real Data ===
def load_real_data():
    X, Y = [], []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.mpg'):
            video_path = os.path.join(DATA_DIR, fname)
            align_path = os.path.join(ALIGN_DIR, fname.replace('.mpg', '.align'))
            if not os.path.exists(align_path):
                continue
            video = preprocess_video(video_path)
            labels = align_to_char_sequence(align_path)

            if len(labels) < MAX_FRAMES:
                labels += [40] * (MAX_FRAMES - len(labels))
            elif len(labels) > MAX_FRAMES:
                labels = labels[:MAX_FRAMES]

            X.append(video)
            Y.append(labels)
    return np.array(X), np.array(Y)

# === Train and Save ===
def train_and_save_model():
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("📥 Loading real video + alignment data...")
    X_train, y_train = load_real_data()
    print(f"✅ Loaded {len(X_train)} training samples")

    model.fit(X_train, y_train, epochs=15, batch_size=2)

    os.makedirs('models', exist_ok=True)
    model.save_weights('models/complete_model.h5')
    print("✅ Model trained and saved as models/complete_model.h5")

# === Load Model for Inference ===
def load_model():
    model = create_model()
    model.load_weights('models/complete_model.h5')
    print("✅ Model weights loaded from models/complete_model.h5")
    return model

if __name__ == "__main__":
    train_and_save_model()
