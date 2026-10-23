from load_data import load_images, flatten_images, scale_images, split_data
from pathlib import Path
import numpy as np
from sklearn.svm import LinearSVC
import time

images_path = Path("data/images.npy")
labels_path = Path("data/labels.npy")
data_dir = Path("data/EuroSAT_RGB")

if images_path.exists() and labels_path.exists():
    images = np.load(images_path)
    labels = np.load(labels_path)
else:
    images, labels = load_images(data_dir)
    np.save(images_path, images)
    np.save(labels_path, labels)
    
images = flatten_images(images)
images = scale_images(images)
X_train, X_test, y_train, y_test = split_data(images, labels)
print("Train:", X_train.shape)
print("Test:", X_test.shape)

model = LinearSVC()
start = time.time()
model.fit(X_train[:1000], y_train[:1000])
end = time.time()
print("Training took", round(end - start, 1), "seconds")
print("Accuracy:", model.score(X_test, y_test))