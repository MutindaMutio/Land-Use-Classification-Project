import numpy as np
import torch
from torch.utils.data import DataLoader
from load_data import split_data
from dataset import EuroSATDataset
from model import EuroSATCNN

images = np.load("data/images.npy")
labels = np.load("data/labels.npy")

_, X_test, _, y_test = split_data(images, labels)

test_dataset = EuroSATDataset(X_test, y_test)
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
        all_preds.append(preds.cpu().numpy())

y_pred = np.concatenate(all_preds)

print(len(y_pred))
print((y_pred == y_test).mean())