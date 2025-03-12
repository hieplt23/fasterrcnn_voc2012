# Faster R-CNN Object Detection on Pascal VOC 2012

This project implements a Faster R-CNN model for object detection using the Pascal VOC 2012 dataset. The model is built using PyTorch and Streamlit is used for the web interface to facilitate easy interaction with the model.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Contributing](#contributing)

## Project Overview

Faster R-CNN is a state-of-the-art object detection model that is capable of detecting multiple objects in an image with high accuracy. This project leverages the Pascal VOC 2012 dataset to train and evaluate the model. The web interface allows users to upload images or use a webcam for real-time object detection.

## Features

- **Image Upload**: Upload images in JPG, PNG, or JPEG format for object detection.
- **Real-Time Detection**: Use your webcam for real-time object detection.
- **Download Results**: Download the processed images with detected objects.
- **Customizable Settings**: Adjust image size and confidence threshold for detection.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/fasterrcnn_pascalvoc2012.git
   cd fasterrcnn_pascalvoc2012
   ```

2. **Install dependencies**:
   Ensure you have Python 3.8+ and pip installed. Then, run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   Download the Pascal VOC 2012 dataset and place it in the `data/VOCdevkit/VOC2012` directory.

## Usage

To run the application, execute the following command:
```bash
streamlit run app_v1.py
```

This will start a local server, and you can access the application in your web browser.

### Command-Line Arguments

- `--image_path`: Path to the test image.
- `--image_size`: Size of the image for processing.
- `--num_classes`: Number of classes in the dataset.
- `--checkpoint_path`: Path to the trained model checkpoint.

## Dataset

The Pascal VOC 2012 dataset is used for training and evaluation. It contains images with annotations for 20 different object classes.

## Model Training

The model is trained using the Faster R-CNN architecture with a ResNet-50 backbone. Training scripts and configurations are provided in the repository.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.
