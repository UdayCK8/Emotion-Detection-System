# Emotion-Detection-System
Real-time facial emotion detection system using a CNN trained on grayscale images to classify 5 emotions (Angry, Fear, Happy, Sad, Surprise). Built with TensorFlow/Keras for model training, OpenCV for face detection and webcam capture, and Flask for serving live predictions through a web interface.


# Real-Time Facial Emotion Detection System

A real-time facial emotion detection system that uses a Convolutional Neural Network (CNN) to classify human emotions from webcam video. The model is trained on grayscale facial images and served through a Flask web application with live video streaming.

## Overview

This project detects faces in a live webcam feed using OpenCV, classifies the detected facial expression using a trained CNN, and overlays the predicted emotion directly on the video stream in real time — all accessible through a simple web interface.

## Features

- Real-time webcam-based face detection (OpenCV Haar Cascade)
- CNN-based emotion classification across 5 classes: **Angry, Fear, Happy, Sad, Surprise**
- Live MJPEG video streaming through a Flask web app
- Class-weighted training to handle dataset imbalance
- Data augmentation (rotation, shift, zoom, shear, flip) for better generalization
- Early stopping and learning rate scheduling for stable training
- Confidence score displayed alongside predicted emotion

## Tech Stack

| Component | Technology |
|---|---|
| Model Training | TensorFlow / tf-keras |
| Face Detection | OpenCV (Haar Cascade) |
| Web Framework | Flask |
| Image Processing | OpenCV, NumPy, PIL |
| Data Augmentation | tf-keras ImageDataGenerator |

## Project Structure

```
Emotion Detection System/
├── app.py                  # Flask app with real-time webcam detection
├── train_model.py          # CNN training script
├── templates/
│   └── index.html          # Web UI for live video feed
├── model/
│   ├── emotion_cnn.h5       # Final trained model
│   └── best_emotion_cnn.h5  # Best checkpoint during training
├── dataset/                 # Not included — see Dataset section
│   ├── train/
│   │   ├── angry/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/
│       ├── angry/
│       ├── fear/
│       ├── happy/
│       ├── sad/
│       └── surprise/
├── requirements.txt
└── README.md
```

## Dataset

This project expects a FER-style dataset (e.g. [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)) organized into folders by class, as shown in the Project Structure above. Each class folder should contain grayscale facial images sized 48x48 pixels (images are automatically resized during loading if they aren't already this size).

Download a dataset, restructure it to match the layout above, and place it in the project root under `dataset/`.

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/UdayCK8/Emotion-Detection-System.git
   cd Emotion-Detection-System
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv emotion-env
   emotion-env\Scripts\activate      # Windows
   source emotion-env/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add the dataset** following the structure described above.

## Training the Model

```bash
python train_model.py
```

This trains the CNN on the data in `dataset/train` and validates against `dataset/test`. The best-performing checkpoint is saved to `model/best_emotion_cnn.h5`, and the final model is saved to `model/emotion_cnn.h5`.

## Running the Web App

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser. The page displays a live webcam feed with detected faces boxed and labeled with the predicted emotion and confidence score.

## Model Architecture

A CNN with four convolutional blocks (32 → 64 → 128 → 256 filters), each followed by batch normalization, max pooling, and dropout for regularization, then dense layers leading to a 5-class softmax output.

## Future Improvements

- Expand to additional emotion classes (e.g. neutral, disgust)
- Replace Haar Cascade with a more robust deep learning–based face detector (e.g. MTCNN or MediaPipe)
- Add model quantization for faster inference on edge devices
- Deploy as a containerized service (Docker) for easier hosting

## License

This project is for academic and educational purposes.