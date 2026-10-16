# Satellite Image Land-Use Classifier

This project takes a 64x64 satellite image and predicts its land-use category, such as forest, river, or residential. It uses the [EuroSAT dataset](https://zenodo.org/records/7711810).

**Status:** In progress. The repo is set up and the dataset has been chosen; no models have been trained yet.

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