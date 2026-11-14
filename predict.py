import sys
import numpy as np
from PIL import Image
from load_data import prepare_batch, get_class_names
from pathlib import Path
import torch
from model import EuroSATCNN

if len(sys.argv) != 2:
    print("Usage: python predict.py path/to/image")
    sys.exit(1)

# sys.argv is a list like ["predict.py", "some/image.jpg"]
image_path = sys.argv[1]
# convert("RGB") always gives 3 channels of uint8 0 to 255, the same format as the training cache
image = np.array(Image.open(image_path).convert("RGB"))

# The final Linear layer only fits 64x64 input, so fail early with a clear message
if image.shape != (64, 64, 3):
    print(f"Expected a 64x64 image, got {image.shape[1]}x{image.shape[0]}")
    sys.exit(1)

# Add a batch dimension of 1; the model expects a batch, and prepare_batch does permute and / 255
batch = prepare_batch(image[np.newaxis])

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EuroSATCNN().to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

with torch.no_grad():
    outputs = model(batch.to(device))

predicted = outputs.argmax(dim=1).item()
class_names = get_class_names(Path("data/EuroSAT_RGB"))
print("Predicted:", class_names[predicted])