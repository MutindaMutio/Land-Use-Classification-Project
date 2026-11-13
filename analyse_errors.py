import numpy as np
import torch
from torch.utils.data import DataLoader
from load_data import split_data, get_class_names
from dataset import EuroSATDataset
from model import EuroSATCNN
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

images = np.load("data/images.npy")
labels = np.load("data/labels.npy")

_, X_test, _, y_test = split_data(images, labels)

test_dataset = EuroSATDataset(X_test, y_test)
# shuffle=False so predictions line up with y_test by position
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EuroSATCNN().to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

all_preds = []
with torch.no_grad():
    for images_batch, _ in test_loader:
        images_batch = images_batch.to(device)
        outputs = model(images_batch)
        preds = outputs.argmax(dim=1)
        # .cpu() first because NumPy can't read a tensor that lives on the GPU
        all_preds.append(preds.cpu().numpy())

y_pred = np.concatenate(all_preds)

print(len(y_pred))
print((y_pred == y_test).mean())

class_names = get_class_names(Path("data/EuroSAT_RGB"))

# Mean of True/False is the fraction correct, so this is per-class accuracy (recall)
for number, name in enumerate(class_names):
    mask = y_test == number
    correct = y_pred[mask] == y_test[mask]
    print(name, round(correct.mean(), 3))

# Rows are the true class, columns the predicted class, both in class-number order
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 6 is PermanentCrop and 2 is HerbaceousVegetation, the largest single confusion in the matrix
wrong = (y_test == 6) & (y_pred == 2)
indices = np.where(wrong)[0]
right = (y_test == 6) & (y_pred == 6)
right_indices = np.where(right)[0]

def show_images(image_numbers, title):
    # Only the first 10, in test-set order (not random) so these are not a representative sample
    fig, axes = plt.subplots(2, 5)
    fig.suptitle(title)
    for ax, number in zip(axes.flatten(), image_numbers[:10]):
        ax.imshow(X_test[number])
        ax.axis("off")

show_images(indices, "PermanentCrop called HerbaceousVegetation")
show_images(right_indices, "PermanentCrop called PermanentCrop (correct)")

# Average colour: axis=(0, 1, 2) averages over images, height and width, leaving one value each for R, G, B
herb = y_test == 2
print("Wrong:", X_test[indices].mean(axis=(0, 1, 2)).round(1))
print("Correct:", X_test[right_indices].mean(axis=(0, 1, 2)).round(1))
print("HerbaceousVegetation:", X_test[herb].mean(axis=(0, 1, 2)).round(1))

# One show() at the end so both image windows open together; it blocks until they are closed
plt.show()
