import torch
import numpy as np

class EuroSATDataset:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Convert one image at a time so the cache stays small uint8 (about 332 MB) instead of a float array of about 2.65 GB
        image = torch.from_numpy(self.images[index])
        # NumPy stores (height, width, channels); PyTorch expects channels first. float32 matches the model, / 255 scales pixels to 0 to 1
        image = image.permute(2, 0, 1).float() / 255
        return image, self.labels[index]
    
if __name__ == "__main__":
    # Check on fake data; should print 3 torch.Size([3, 64, 64]) 1
    fake_images = np.zeros((3, 64, 64, 3), dtype=np.uint8)
    fake_labels = np.array([0, 1, 2])
    ds = EuroSATDataset(fake_images, fake_labels)
    image, label = ds[1]
    print(len(ds), image.shape, label)