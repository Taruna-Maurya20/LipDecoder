# LipDecoder



# 💬 LipBuddy – Deep Learning-Based Lip Reading Model

**LipBuddy** is a deep learning-based lip reading system that converts silent video of spoken words into text using computer vision and sequence modeling techniques. The project is based on the **LipNet** architecture and enhances visual speech recognition by using 3D convolutional networks and bidirectional LSTMs, deployed using **Streamlit** for interactive demonstrations.

---

## 📜 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [License](#license)

---

## 🧠 Introduction

LipBuddy aims to interpret speech from silent video by analyzing lip movements. The project leverages computer vision and deep learning (Conv3D + BiLSTM + CTC Loss) to create an end-to-end system for translating visual information into accurate text output.

This can be especially useful for applications like:
- Assisting hearing-impaired individuals
- Surveillance and silent communication
- Enhancing voice recognition in noisy environments

---

## ✨ Features

- Lip reading from silent `.mpg` videos
- Conv3D + BiLSTM architecture
- CTC loss for alignment-free transcription
- Streamlit-based UI for real-time testing
- Modular and extensible codebase

---

## 🧱 Architecture

The model is composed of the following key components:

- **Conv3D Layers**: For spatiotemporal feature extraction from video frames
- **TimeDistributed Flattening**: Converts 3D feature maps into sequences
- **BiLSTM Layers**: Captures temporal dependencies in both directions
- **CTC Loss**: Enables training without needing frame-wise alignment
- **Softmax Output**: Maps predictions to a predefined character set

---

## 📁 Dataset

We used the **GRID Corpus**, a dataset of audiovisual sentences spoken by multiple speakers. Each sample consists of:

- 75 frames per video
- 46×140 grayscale image resolution
- Aligned text labels for each video

> Note: Data preprocessing includes frame extraction, resizing, and normalization.

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/LipBuddy.git
   cd LipBuddy






*Create a virtual environment (optional but recommended):
python -m venv env
source env/bin/activate  # or .\env\Scripts\activate on Windows




Install dependencies:
pip install -r requirements.txt




Train the model: python train.py
Run streamlit app: streamlit run app.py

