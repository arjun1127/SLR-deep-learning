# Sign Language Recognition using Deep Learning and LSTM

This project implements a sign language recognition system using deep learning, specifically Long Short-Term Memory (LSTM) networks, combined with MediaPipe's Holistic detection to track and recognize hand gestures.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

This project aims to recognize sign language gestures using deep learning techniques. The system is built with TensorFlow and Keras for the deep learning model, and MediaPipe Holistic for detecting and tracking hand and body movements.

## Features

- Real-time sign language recognition
- Utilizes LSTM networks for sequence prediction
- Uses MediaPipe Holistic for accurate hand, face, and pose tracking

## Project Structure

```plaintext
.
├── dataset/
│   ├── MP_DATA/
├── models/
│   ├── action1.h5
│   ├── action2.h5
├── src/
│   ├── main.py
│   ├── train.py
│   └── utils.py
├── MP_DATA_FINAL/  # Stored keypoints as numpy arrays
├── requirements.txt
└── README.md

