from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def get_class_names(data_dir):
    class_names = []
    for class_folder in data_dir.iterdir():
        if class_folder.is_dir():
            class_names.append(class_folder.name)
    return sorted(class_names)

def load_images(data_dir, show_progress=True):
    images = []
    labels = []
    class_names = get_class_names(data_dir)
    for number, name in enumerate(class_names):
        if show_progress:
            print("loading", name)
        files = sorted((data_dir / name).iterdir())
        for image_path in files:
            images.append(plt.imread(image_path))
            labels.append(number)
    labels = np.array(labels)
    images = np.array(images)
    return images, labels

def flatten_images(images):
    return images.reshape(len(images), -1)

def scale_images(images):
    return images / 255

def split_data(images, labels):
    return train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

if __name__ == "__main__":
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