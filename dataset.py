import torch
import numpy as np

class EuroSATDataset:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index])
        image = image.permute(2, 0, 1).float() / 255
        return image, self.labels[index]
    
if __name__ == "__main__":
    fake_images = np.zeros((3, 64, 64, 3), dtype=np.uint8)
    fake_labels = np.array([0, 1, 2])
    ds = EuroSATDataset(fake_images, fake_labels)
    image, label = ds[1]
    print(len(ds), image.shape, label)