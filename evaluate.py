import numpy as np
import torch
from torch.utils.data import DataLoader
from load_data import split_data
from dataset import EuroSATDataset
from model import EuroSATCNN

images = np.load("data/images.npy")
labels = np.load("data/labels.npy")

# Same split and random_state as the baseline, so both are scored on the same 5,400 test images
X_train, X_test, y_train, y_test = split_data(images, labels)

test_dataset = EuroSATDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
# The file holds only the weights, so build the model first; map_location lets this load without a GPU
model = EuroSATCNN().to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
# No dropout or batch-norm in this model, so eval() changes nothing currently, but it keeps this safe if either is added
model.eval()

correct = 0
# Nothing is being trained, so skip gradient tracking
with torch.no_grad():
    for images_batch, labels_batch in test_loader:
        images_batch = images_batch.to(device)
        labels_batch = labels_batch.to(device)

        outputs = model(images_batch)
        correct += (outputs.argmax(dim=1) == labels_batch).sum().item()

# Describes whichever best_model.pt exists now; retraining overwrites it, so this number would no longer match
print(f"Test accuracy: {correct / len(test_dataset):.3f}")