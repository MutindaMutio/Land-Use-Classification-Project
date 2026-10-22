from load_data import load_images, flatten_images, scale_images, split_data
from pathlib import Path

data_dir = Path("data/EuroSAT_RGB")
images, labels = load_images(data_dir)
images = flatten_images(images)
images = scale_images(images)
X_train, X_test, y_train, y_test = split_data(images, labels)
print("Train:", X_train.shape)
print("Test:", X_test.shape)

