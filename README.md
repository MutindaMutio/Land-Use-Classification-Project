# Satellite Image Land-Use Classifier

This project takes a 64x64 satellite image and predicts its land-use category, such as forest, river, or residential. It uses the [EuroSAT dataset](https://zenodo.org/records/7711810).

**Status:** In progress. The data has been downloaded and explored; no models have been trained yet.

**Planned:** A simple scikit-learn baseline and a PyTorch CNN, then comparing the two. 

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
5. Install packages in requirements.txt
    - ```pip install -r requirements.txt```
6. Run download_data.py
    - ```python download_data.py```
    - Takes ~10 mins to download data
7. Run explore_data.py
    - ```python explore_data.py```
    - Opens a window that needs to be closed to finish the script

## Results

The baseline is a linear classifier (SGDClassifier on default settings) trained on raw pixels. It scored 35.7% accuracy on the 5,400 test images, compared with about 10% when randomly guessing.

## Citation
Helber, P., Bischke, B., Dengel, A., & Borth, D. (2018). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. Zenodo. https://doi.org/10.5281/zenodo.7711810