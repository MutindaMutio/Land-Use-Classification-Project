import numpy as np
from load_data import split_data, split_validation
from torch.utils.data import DataLoader
from dataset import EuroSATDataset
import torch
from model import EuroSATCNN
import time

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

num_epochs = 5
for epoch in range(num_epochs):
    start = time.time()

    model.train()
    total_loss = 0
    for images_batch, labels_batch in train_loader:
        images_batch = images_batch.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(images_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images_batch, labels_batch in val_loader:
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(images_batch)
            val_loss += criterion(outputs, labels_batch).item()
            correct += (outputs.argmax(dim=1) == labels_batch).sum().item()

    print(f"Epoch {epoch + 1}: train loss {total_loss / len(train_loader):.3f}, val loss {val_loss / len(val_loader):.3f}, val acc {correct / len(val_dataset):.3f}, {time.time() - start:.0f}s")