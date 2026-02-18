import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Suppress TensorFlow logs and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN warnings

def load_dataset(folder_path, image_size=(128, 128)):
    """
    Load images from folders where each folder represents a blood group.

    Args:
        folder_path (str): Path to the root dataset folder.
        image_size (tuple): Size to resize the images (width, height).

    Returns:
        images (numpy.ndarray): Array of preprocessed images.
        labels (numpy.ndarray): Corresponding labels for the images.
        label_map (dict): Mapping of blood group names to numeric labels.
    """
    images = []
    labels = []
    label_map = {}
    current_label = 0

    for folder_name in sorted(os.listdir(folder_path)):
        folder_full_path = os.path.join(folder_path, folder_name)
        if not os.path.isdir(folder_full_path):
            continue

        label_map[folder_name] = current_label
        for file_name in os.listdir(folder_full_path):
            file_path = os.path.join(folder_full_path, file_name)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Skipping invalid image {file_path}")
                continue
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(current_label)
        current_label += 1

    if not images:
        raise ValueError("No images found in the dataset. Please check the folder structure.")

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int32)
    return images, labels, label_map

def build_cnn_model(input_shape, num_classes):
    """
    Build and compile a CNN model.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.

    Returns:
        model: Compiled CNN model.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Path to your dataset folder
    dataset_path = "dataset"  # Ensure this folder exists in the same directory as this script
    image_size = (128, 128)

    # Load the dataset
    print("Loading dataset...")
    images, labels, label_map = load_dataset(dataset_path, image_size)
    print(f"Loaded {len(images)} images.")
    print(f"Label Map: {label_map}")

    # Preprocess data
    images = images.reshape(-1, image_size[0], image_size[1], 1)  # Reshape for grayscale
    labels = to_categorical(labels, num_classes=len(label_map))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Build and train the model
    print("Building and training the model...")
    model = build_cnn_model(input_shape=(image_size[0], image_size[1], 1), num_classes=len(label_map))
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Save the trained model
    model.save("blood_group_model.h5")
    print("Model saved as blood_group_model.h5")
