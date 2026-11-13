import torch
from torch import nn

class EuroSATCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # padding=1 with kernel_size=3 keeps height and width unchanged, so only pooling shrinks the image
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Pooling has no weights, so one layer is reused after both conv layers
        self.pool = nn.MaxPool2d(2)
        # 64x64 is halved twice to 16x16, with 32 channels; 10 is the number of classes. Change the conv or pool layers and this input size must change too
        self.fc = nn.Linear(32 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        # Start flattening at dimension 1 so the batch dimension is kept
        x = torch.flatten(x, 1)
        # Raw scores with no softmax, because CrossEntropyLoss expects raw scores
        return self.fc(x)

if __name__ == "__main__":
    # Shape check on a fake batch; should print torch.Size([5, 10])
    model = EuroSATCNN()
    x = torch.zeros(5, 3, 64, 64)
    print(model(x).shape)