# Project Requirements & Setup Guide

This document explains all required modules and how to install them for the project.

---

## 📦 Required Modules

### 1️⃣ TensorFlow
Used for building and training the neural network.

```bash
pip install tensorflow
```

> ✅ TensorFlow supports GPU automatically if properly configured.

---

### 2️⃣ NumPy
Used for numerical computations and array manipulation.

```bash
pip install numpy
```

---

### 3️⃣ OpenCV
Used for image preprocessing and manipulation.

```bash
pip install opencv-python
```

---

### 4️⃣ Scikit-learn
Used for splitting datasets into training and testing sets.

```bash
pip install scikit-learn
```

---

### 5️⃣ Flask (Optional – For Deployment)
Used for creating a backend API if deploying as a web application.

```bash
pip install flask
```

---

### 6️⃣ Matplotlib (Optional)
Used for visualizing training progress (loss, accuracy graphs).

```bash
pip install matplotlib
```

---

# 🚀 Combined Installation Command

You can install all required packages at once:

```bash
pip install tensorflow numpy opencv-python scikit-learn flask matplotlib
```

---

# ⚡ GPU Support Setup (Optional)

If you have a compatible NVIDIA GPU, follow these steps:

### 1️⃣ Install NVIDIA GPU Drivers
Download the latest drivers from the official NVIDIA website.

### 2️⃣ Install CUDA Toolkit
Download and install the CUDA Toolkit.

### 3️⃣ Install cuDNN
Download and install cuDNN compatible with your CUDA version.

---

### ✅ Verify GPU Installation

Run the following command to check if TensorFlow detects your GPU:

```bash
python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
```

If configured correctly, your GPU should appear in the output.

---

# 🧪 Verify Installation

Run this Python script to ensure all modules are installed correctly:

```python
import tensorflow as tf
import numpy as np
import cv2
import sklearn
import flask
import matplotlib

print("All modules installed successfully!")
print("TensorFlow version:", tf.__version__)
```

---

# 📌 Notes

- Make sure you are using a virtual environment (recommended).
- Ensure CUDA and cuDNN versions are compatible with your TensorFlow version.
- Use `python --version` and `pip --version` to verify your environment.

---

If you face any installation issues, feel free to raise an issue in this repository.
