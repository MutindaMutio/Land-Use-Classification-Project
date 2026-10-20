from load_data import load_images, flatten_images, scale_images
from pathlib import Path

data_dir = Path("data/EuroSAT_RGB")
images, labels = load_images(data_dir)
images = flatten_images(images)
images = scale_images(images)
print(images.shape)

