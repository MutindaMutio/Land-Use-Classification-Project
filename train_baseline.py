from load_data import load_images, flatten_images, scale_images, split_data, get_class_names
from pathlib import Path
import numpy as np
from sklearn.linear_model import SGDClassifier
import time

images_path = Path("data/images.npy")
labels_path = Path("data/labels.npy")
data_dir = Path("data/EuroSAT_RGB")
class_names = get_class_names(data_dir)

# If both cache files exist, load them (seconds) instead of re-reading all 27,000 image files (minutes)
if images_path.exists() and labels_path.exists():
    images = np.load(images_path)
    labels = np.load(labels_path)
else:
    # No cache yet: load the images the slow way and save them so the next run is faster
    images, labels = load_images(data_dir)
    np.save(images_path, images)
    np.save(labels_path, labels)
    
images = flatten_images(images)
images = scale_images(images)
X_train, X_test, y_train, y_test = split_data(images, labels)
print("Train:", X_train.shape)
print("Test:", X_test.shape)

model = SGDClassifier(random_state=42)
start = time.time()
model.fit(X_train, y_train)
end = time.time()
print("Training took", round(end - start, 1), "seconds")
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))
y_pred = model.predict(X_test)
print("Predictions:", y_pred.shape)

for number, name in enumerate(class_names):
    mask = y_test == number
    correct = y_pred[mask] == y_test[mask]
    print(name, round(correct.mean(), 3))