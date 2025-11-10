# TamperNet - Image Tampering Detection using Deep Learning and Forensic Feature Analysis

# Overview

This project focuses on detecting tampered digital images using Deep Learning and forensic feature analysis. The system implements a Dual Branch CNN architecture that extracts both global semantic information and low-level forensic cues to identify image manipulations. A pre-trained MobileNet-V2 backbone is utilized for feature representation, while noise inconsistencies, compression patterns, and edge artifacts are leveraged for forensic signal strengthening.

This model achieved 93.6% accuracy on 12,615 test samples.

# Features

. Dual Branch CNN based tampering detection

. Global semantic + forensic feature fusion

. Tampering localization visualization (heatmaps)

. Custom dataset fine-tuning support

. High performance on benchmark tampering datasets

# Tech Stack

Category	Tools

. Deep Learning :	TensorFlow, Keras, PyTorch

. Computer Vision	: OpenCV, scikit-image

. Machine Learning	: scikit-learn

. Data Processing :	NumPy, Pandas

. Visualization	: Matplotlib, Seaborn

# Datasets

Due to size and licensing restrictions, datasets (CASIA V2, CASIA CMFD) are not included in this repository. Users must download them from their official sources.

Dataset	Source
CASIA V2 :	[http://forensics.idealtest.org/](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset)

CASIA CMFD :	[CASIA-CMFD](https://www.kaggle.com/datasets/mashraffarouk/casia-cmfd)

# Preprocessing Instructions

. Normalize pixel values to [0,1]

. Resize images to the MobileNet-V2 input size

. Store labels clearly (Authentic vs Tampered)

. Recommended augmentations: random crop, flip, rotate, gaussian noise

# Repository Contents

. Dual Branch CNN architecture implementation

. Training scripts

. Inference / prediction script

. Heatmap visualization module

. Experiment configuration files / notebooks

# Objective

To enhance digital forensics by combining deep learning with forensic feature analysis to deliver interpretable, reliable, and accurate image tampering detection for research and investigation scenarios.

# Future Work

. Multi-region tampering localization

. Multi-modal forgery detection

. Transformer-based forensic reasoning

. Adversarial robustness evaluation
