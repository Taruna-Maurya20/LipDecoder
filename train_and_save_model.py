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




# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv3D, MaxPool3D, Activation, TimeDistributed, Flatten, Bidirectional, LSTM, Dropout, Dense

# # === Constants ===
# DATA_DIR = r"C:\Users\Abhilasha\Desktop\lipbuddy\data\s1"
# ALIGN_DIR = r"C:\Users\Abhilasha\Desktop\lipbuddy\data\alignments\s1"
# IMG_HEIGHT = 46
# IMG_WIDTH = 140
# MAX_FRAMES = 75
# NUM_CLASSES = 41  # 0–40 including blank

# # === Character Map ===
# CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
# PHONES = ['CH', 'SH', 'TH', 'EE', 'OO', 'AY', 'AR', 'ER', 'OW', 'UH', 'IH', 'B', 'D']
# ALL_LABELS = CHARS + PHONES
# CHAR_MAP = {ch: idx for idx, ch in enumerate(ALL_LABELS)}
# CHAR_MAP[''] = 40  # blank class

# # === Model ===
# def create_model():
#     model = Sequential()
#     model.add(tf.keras.Input(shape=(MAX_FRAMES, IMG_HEIGHT, IMG_WIDTH, 1)))

#     model.add(Conv3D(128, 3, padding='same', activation='relu'))
#     model.add(MaxPool3D((1, 2, 2)))

#     model.add(Conv3D(256, 3, padding='same', activation='relu'))
#     model.add(MaxPool3D((1, 2, 2)))

#     model.add(Conv3D(75, 3, padding='same', activation='relu'))
#     model.add(MaxPool3D((1, 2, 2)))

#     model.add(TimeDistributed(Flatten()))
#     model.add(Bidirectional(LSTM(128, return_sequences=True)))
#     model.add(Dropout(0.5))
#     model.add(Bidirectional(LSTM(128, return_sequences=True)))
#     model.add(Dropout(0.5))

#     model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
#     return model

# # === Preprocess Video ===
# def preprocess_video(path):
#     cap = cv2.VideoCapture(path)
#     frames = []

#     while len(frames) < MAX_FRAMES:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
#         frames.append(resized)
#     cap.release()

#     if len(frames) < MAX_FRAMES:
#         frames += [np.zeros((IMG_HEIGHT, IMG_WIDTH))] * (MAX_FRAMES - len(frames))
#     elif len(frames) > MAX_FRAMES:
#         frames = frames[:MAX_FRAMES]

#     video = np.array(frames)[..., np.newaxis] / 255.0
#     return video

# # === Read Alignment File ===
# def align_to_char_sequence(filepath):
#     with open(filepath, 'r') as f:
#         words = [line.strip().split()[-1] for line in f.readlines()]
#     sentence = ' '.join(words)
#     return [CHAR_MAP.get(ch.upper(), 40) for ch in list(sentence) if ch.upper() in CHAR_MAP or ch == ' ']

# # === Load Real Data ===
# def load_real_data():
#     X, Y = [], []
#     for fname in os.listdir(DATA_DIR):
#         if fname.endswith('.mpg'):
#             video_path = os.path.join(DATA_DIR, fname)
#             align_path = os.path.join(ALIGN_DIR, fname.replace('.mpg', '.align'))
#             if not os.path.exists(align_path):
#                 continue
#             video = preprocess_video(video_path)
#             labels = align_to_char_sequence(align_path)

#             if len(labels) < MAX_FRAMES:
#                 labels += [40] * (MAX_FRAMES - len(labels))
#             elif len(labels) > MAX_FRAMES:
#                 labels = labels[:MAX_FRAMES]

#             X.append(video)
#             Y.append(labels)
#     return np.array(X), np.array(Y)

# # === Train and Save (with Resume Option) ===
# def train_and_save_model():
#     model = create_model()
#     weights_path = 'models/complete_model_epoch_resume.h5'

#     if os.path.exists(weights_path):
#         model.load_weights(weights_path)
#         print(f"🔁 Resumed training from: {weights_path}")
#     else:
#         print("🆕 Starting fresh training")

#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     print("📥 Loading real video + alignment data...")
#     X_train, y_train = load_real_data()
#     print(f"✅ Loaded {len(X_train)} training samples")

#     model.fit(X_train, y_train, epochs=2, batch_size=2)

#     os.makedirs('models', exist_ok=True)
#     model.save_weights(weights_path)
#     print(f"✅ Model weights saved to {weights_path}")

# # === Load Trained Model for Inference ===
# def load_model():
#     model = create_model()
#     model.load_weights('models/complete_model_epoch_resume.h5')
#     print("✅ Model weights loaded from saved file")
#     return model

# if __name__ == "__main__":
#     train_and_save_model()

                                                                                                                                                     