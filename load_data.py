from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch 

def get_class_names(data_dir):
    class_names = []
    for class_folder in data_dir.iterdir():
        if class_folder.is_dir():
            class_names.append(class_folder.name)
    # Sorted because iterdir() order is not guaranteed; a class's number is its position in this list
    return sorted(class_names)

def load_images(data_dir, show_progress=True):
    images = []
    labels = []
    class_names = get_class_names(data_dir)
    for number, name in enumerate(class_names):
        if show_progress:
            print("loading", name)
        # Sorted so the image order, and so the labels, is the same on every machine
        files = sorted((data_dir / name).iterdir())
        for image_path in files:
            # imread gives uint8 values 0 to 255 for these files; scale_images and the Dataset both assume that
            images.append(plt.imread(image_path))
            labels.append(number)
    labels = np.array(labels)
    images = np.array(images)
    return images, labels

def flatten_images(images):
    # -1 lets NumPy work out 12,288 (64*64*3), so the number isn't hard-coded
    return images.reshape(len(images), -1)

def scale_images(images):
    return images / 255

def split_data(images, labels):
    # stratify keeps each class at 20% of the test set, which matters because the arrays are stored in class order; random_state=42 gives the same split every run, so the baseline and the CNN share the same test images
    return train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

def split_validation(X_train, y_train):
    # 25% of the remaining 80% is 20% of the total, which gives a 60/20/20 split
    return train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)

def prepare_batch(images):
    t = torch.from_numpy(images)
    t = t.permute(0, 3, 1, 2)
    return t.float() / 255

if __name__ == "__main__":
    # Manual check of the pipeline; it decodes all 27,000 images, so it is slow
    data_dir = Path("data/EuroSAT_RGB")
    names = get_class_names(data_dir)
    print(names)
    images, labels = load_images(data_dir)
    images = flatten_images(images)
    images = scale_images(images)
    print(images.shape)
    print(np.bincount(labels))
    X_train, X_test, y_train, y_test = split_data(images, labels)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(np.bincount(y_test))