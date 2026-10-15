from pathlib import Path

data_dir = Path("data/EuroSAT_RGB")

for class_folder in data_dir.iterdir():
    images = list(class_folder.iterdir())
    print(class_folder.name, len(images))
