from pathlib import Path

def get_class_names(data_dir):
    class_names = []
    for class_folder in data_dir.iterdir():
        if class_folder.is_dir():
            class_names.append(class_folder.name)
    return sorted(class_names)
if __name__ == "__main__":
    data_dir = Path("data/EuroSAT_RGB")
    names = get_class_names(data_dir)
    print (names)

  