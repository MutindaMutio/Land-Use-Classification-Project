from pathlib import Path
import matplotlib.pyplot as plt

data_dir = Path("data/EuroSAT_RGB")

for class_folder in data_dir.iterdir():
    images = list(class_folder.iterdir())
    first_image = plt.imread(images[0])
    print(class_folder.name, len(images), first_image.shape)
    print("  min:", first_image.min(), "max:", first_image.max(), "type:", first_image.dtype)