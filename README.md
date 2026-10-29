# Satellite Image Land-Use Classifier

This project takes a 64x64 satellite image and predicts its land-use category, such as forest, river, or residential. It uses the [EuroSAT dataset](https://zenodo.org/records/7711810).

**Status:** In progress. scikit-learn baseline model trained (35.7% test accuracy).

**Planned:** A PyTorch CNN, then comparing the two.

## Dataset

EuroSAT has 27,000 labelled images in 10 classes:

- `AnnualCrop`: 3000 images
- `Forest`: 3000 images
- `HerbaceousVegetation`: 3000 images
- `Highway`: 2500 images
- `Industrial`: 2500 images
- `Pasture`: 2000 images
- `PermanentCrop`: 2500 images
- `Residential`: 3000 images
- `River`: 2500 images
- `SeaLake`: 3000 images

## How to run
1. Clone the repo
    - ```git clone "https://github.com/MutindaMutio/Land-Use-Classification-Project.git"```
2. Move into the project folder
    - ```cd Land-Use-Classification-Project```
3. Create the virtual environment 
    - ```python -m venv .venv```
4. Activate it (Windows)
    - ```.venv\Scripts\activate```
5. Install PyTorch first
    - Go to https://pytorch.org/get-started/locally/ and choose your operating system, Pip, and Compute Platform (a CUDA version, or CPU if you have no NVIDIA GPU), then run the command it shows
    - For reference, this project was built with: ```pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132```
6. Install the other packages in requirements.txt
    - ```pip install -r requirements.txt```
    - requirements.txt was generated with the CUDA 13.2 build of PyTorch, so if you installed a different build in step 5, remove the torch and torchvision lines from it before running this
7. Run download_data.py
    - ```python download_data.py```
    - Takes ~10 mins to download data
8. Run explore_data.py
    - ```python explore_data.py```
    - Opens a window that needs to be closed to finish the script
9. Run train_baseline.py
    - ```python train_baseline.py```
    - Prints overall and per-class test accuracy
    - Takes ~9 mins to train. The first run also decodes the images and caches them in `data/`

## Results

The baseline is a linear classifier (SGDClassifier on default settings) trained on raw pixels. It scored 35.7% accuracy on the 5,400 test images, compared with about 10% when randomly guessing.

Per-class accuracy (share of each class's test images predicted correctly):

| Class | Test images | Accuracy |
|---|---|---|
| AnnualCrop | 600 | 0.638 |
| Forest | 600 | 0.542 |
| HerbaceousVegetation | 600 | 0.007 |
| Highway | 500 | 0.156 |
| Industrial | 500 | 0.614 |
| Pasture | 400 | 0.132 |
| PermanentCrop | 500 | 0.000 |
| Residential | 600 | 0.603 |
| River | 500 | 0.124 |
| SeaLake | 600 | 0.590 |

### Caveats:

- This is a single run (`random_state=42`), with no repeats, so small differences between classes may just be noise.
- Pixels were used unscaled and the classifier has its default settings.
- Training took about 9 minutes on the CPU.
- A first attempt with `LinearSVC` on the full training set ran for over an hour without finishing and was stopped, so `SGDClassifier` was used instead.

## Citation
Helber, P., Bischke, B., Dengel, A., & Borth, D. (2018). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. Zenodo. https://doi.org/10.5281/zenodo.7711810