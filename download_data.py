from pathlib import Path
import urllib.request
import zipfile

# Assumes the script is run from the project root, because "data" is relative to the current working directory 
DATA_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
DATA_DIR = Path("data")
zip_path = DATA_DIR / "EuroSAT_RGB.zip"
extracted_dir = DATA_DIR / "EuroSAT_RGB"
if extracted_dir.exists():
    print("Data already exists.")
else:
    print("Downloading...")
    DATA_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, zip_path)
    print("Download finished.")
    print("Unzipping...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()
    print("Finished unzipping.")