# Human Action Recognition using 3D Convolutional Neural Networks (3D CNN)

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-D00000.svg?logo=Keras)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8.svg?logo=opencv)

## Overview
This repository contains the implementation of a **3D Convolutional Neural Network (3D CNN)** for Human Action Recognition (HAR) from video sequences. Unlike traditional 2D CNN approaches that process frames independently and lose temporal context, this 3D CNN architecture simultaneously extracts spatial (appearance) and temporal (motion) features using 3D kernels. 

This project aims to demonstrate an end-to-end deep learning pipeline capable of achieving high accuracy and near real-time inference speeds under constrained computational resources.

## Key Innovations & Improvements
This project significantly refactors and improves upon baseline hybrid models (e.g., CNN + ConvLSTM2D) by introducing:
1. **Random Continuous Sampling**: Replaced standard sequential sampling (taking the first $N$ frames) with a random starting point approach. This acts as a natural temporal data augmentation, preventing the model from overfitting to the introductory segments of videos.
2. **Global Average Pooling 3D**: Replaced the parameter-heavy `Flatten` layer with `GlobalAveragePooling3D`. This drastically reduces the model's footprint and mitigates overfitting risks while retaining crucial spatiotemporal representations.
3. **Optimized Functional Architecture**: Transitioned from a `Sequential` structure to the Keras `Functional API` for a more robust and scalable model design.

## Dataset
The model is trained and evaluated on a focused subset of the **UCF50 Action Recognition Data Set**, specifically chosen to represent diverse motion dynamics:
* `WalkingWithDog`
* `TaiChi`
* `Swing`
* `HorseRace`

*Input dimensions:* `(Batch, 20 Frames, 64 Height, 64 Width, 3 RGB Channels)`

## Results & Benchmarks
The proposed 3D CNN model demonstrates a remarkable performance leap compared to the baseline ConvLSTM architecture.

| Metric | Baseline Model (ConvLSTM) | Proposed Model (3D CNN) | Improvement |
| :--- | :---: | :---: | :---: |
| **Train Accuracy** | 95.20% | **99.36%** | + 4.16% |
| **Validation Accuracy**| 83.56% | **94.87%** | + 11.31% |
| **Test Accuracy** | 76.23% | **90.82%** | **+ 14.59%** |
| **Test Loss** | 0.7749 | **0.3726** | **- 51.90%** |

* **Generalization Gap**: The gap between training and validation loss was significantly narrowed, effectively curing the severe overfitting observed in the baseline model.
* **Inference Speed**: Tested on an NVIDIA Tesla P100, the model achieves an average inference time of **~0.02 seconds per 20-frame clip**, enabling inference-level real-time processing capabilities.

## Repository Structure
```text
 3d-cnn-human-action-recognition
 ┣  assets/          # Contains images and plots for documentation
 ┣  data/            # (Local) Directory for the UCF50 dataset
 ┣  docs/            # Academic reports and project documentation
 ┃ ┗  Bao-cao-mon-hoc.pdf
 ┣  notebooks/       # Jupyter notebooks for experimentation and training
 ┃ ┗  actionreg-with-3dcnn.ipynb
 ┣  README.md        # Project documentation
 ┣  requirements.txt # Dependencies for reproducibility
 ┗  LICENSE          # MIT License
