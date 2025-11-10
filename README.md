# Image Tampering Detection using Deep Learning and Forensic Feature Analysis

# Project Overview

This Deep Learning-based image tampering detection system applies a Dual Branch CNN architecture to simultaneously extract global semantic features and forensic cues, enabling reliable identification of manipulated content in digital images. The system leverages a pre-trained MobileNet-V2 backbone for high-level feature extraction while analyzing noise inconsistencies, edge artifacts, and compression traces to strengthen forensic evidence-based detection. Evaluated across 12,615 test samples, the model achieved 93.6% classification accuracy.

# Features

# Dual Branch CNN based deep analysis

Global semantic + forensic feature extraction

Manipulation visualization and heatmap support

Custom fine-tuning with user datasets

Multi-dataset compatibility (CASIA V2, CASIA-CMFD)

# Tech Stack
Category	Tools
Deep Learning	TensorFlow, Keras, PyTorch
Computer Vision	OpenCV, scikit-image
ML Utilities	scikit-learn, NumPy, Pandas
Visualization	Matplotlib, Seaborn
📂 Supported Datasets

CASIA V2

CASIA CMFD
