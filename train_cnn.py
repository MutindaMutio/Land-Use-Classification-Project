import numpy as np
from load_data import split_data, split_validation
from torch.utils.data import DataLoader
from dataset import EuroSATDataset
import torch
from model import EuroSATCNN

images = np.load("data/images.npy")
labels = np.load("data/labels.npy")

X_train, X_test, y_train, y_test = split_data(images, labels)
# Judges the model during training to make sure the model is improving as training goes on, so the test set stays unseen until the end
X_train, X_val, y_train, y_val = split_validation(X_train, y_train)

train_dataset = EuroSATDataset(X_train, y_train)
val_dataset = EuroSATDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EuroSATCNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


