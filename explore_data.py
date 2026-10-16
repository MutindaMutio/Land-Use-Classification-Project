from pathlib import Path
import matplotlib.pyplot as plt

data_dir = Path("data/EuroSAT_RGB")
sample_images = []
class_names = []
for class_folder in data_dir.iterdir():
    images = list(class_folder.iterdir())
    first_image = plt.imread(images[0])
    sample_images.append(first_image)
    class_names.append(class_folder.name)
    print(class_folder.name, len(images), first_image.shape)
    print("  min:", first_image.min(), "max:", first_image.max(), "type:", first_image.dtype)
fig, axes = plt.subplots(2, 5)
axes = axes.flatten()

for i in range(10):
    axes[i].imshow(sample_images[i])
    axes[i].set_title(class_names[i])
plt.show()