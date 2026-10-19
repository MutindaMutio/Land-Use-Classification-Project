from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == "__main__":
    data_dir = Path("data/EuroSAT_RGB")
    names = get_class_names(data_dir)
    print(names)
    images, labels = load_images(data_dir)
    print(labels.shape)
    print(images.shape)
    print(np.bincount(labels))