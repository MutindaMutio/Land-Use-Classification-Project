import numpy as np
from load_data import split_data, split_validation

images = np.load("data/images.npy")
labels = np.load("data/labels.npy")

X_train, X_test, y_train, y_test = split_data(images, labels)
# Judges the model during training to make sure the model is improving as training goes on, so the test set stays unseen until the end
X_train, X_val, y_train, y_val = split_validation(X_train, y_train)