# Import all dependencies
import streamlit as st
import os
import imageio
import cv2
import numpy as np
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout for Streamlit app
st.set_page_config(layout='wide')

# Sidebar information
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App')

# Get list of available video files
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Create layout columns
col1, col2 = st.columns(2)

if options:
    # Video Processing
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..', 'data', 's1', selected_video)

        # Convert to MP4 format for display
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Render inside the app
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')

        # 1) Load video + alignment words
        video, annotations = load_data(tf.convert_to_tensor(file_path))

        # 2) Convert video frames to grayscale
        gray_video = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video], dtype=np.uint8)

        # Save as GIF to visualize processed frames
        imageio.mimsave('animation.gif', gray_video, fps=10)

        # --- Show GIF first
        st.image('animation.gif', width=400)

        # 3) Load pre-trained model
        st.info('This is the output of the machine learning model as tokens')
        model = load_model()

        # Ensure correct input shape
        gray_video_expanded = np.expand_dims(gray_video, axis=-1)  # (75, 46, 140, 1)
        gray_video_expanded = np.expand_dims(gray_video_expanded, axis=0)  # (1, 75, 46, 140, 1)

        # Model prediction
        yhat = model.predict(gray_video_expanded)

        # Decode output tokens
        decoder = tf.keras.backend.ctc_decode(yhat, [yhat.shape[1]], greedy=True)[0][0].numpy()
        st.text(f"Predicted Tokens: {decoder}")

        # Remove -1 values (CTC blank tokens)
        filtered_tokens = [t for t in decoder[0] if t != -1]

        # Convert tokens to characters
        decoded_text = num_to_char(filtered_tokens)

        # 4) Finally, show the ground truth words from the .align file
        if annotations:
            st.info('Decoded sentences (from alignment file)')
            st.text(" ".join(annotations))
        else:
            st.error("No alignment words found for this video.")
