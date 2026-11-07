# Satellite Image Land-Use Classifier

This project takes a 64x64 satellite image and predicts its land-use category, such as forest, river, or residential. It uses the [EuroSAT dataset](https://zenodo.org/records/7711810).

**Status:** Baseline and CNN both trained and evaluated on the same 5,400 unseen test images.

**Results (single run):** scikit-learn linear baseline 35.7% test accuracy; PyTorch CNN 81.6% test accuracy (chance is roughly 10%).

**Planned:** Look more closely at the CNN to understand where it goes wrong, then clean up the project to make it easier for people to use and read

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

## Approach

### Data split

The 27,000 images were split with stratification (`random_state=42`) so every class keeps the same proportions in each part. A random, class-balanced 20% (5,400 images) was set aside as the test set. The remaining 21,600 were split again, with 25% (5,400) used as a validation set and 75% (16,200) for training. That gives 16,200 training, 5,400 validation and 5,400 test images (60/20/20). The validation set is used to choose the best epoch, and the test set is only scored once at the end.

### Baseline

A linear classifier (`SGDClassifier`, default settings) trained on the raw pixels flattened into 12,288 values per image. It was fitted on all 21,600 non-test images, so it saw more training data than the CNN did.

### CNN

A small convolutional network (`model.py`) with two convolution blocks and one linear layer:

| Layer | Output shape |
|---|---|
| Input | 3 x 64 x 64 |
| Conv (3 to 16 channels, 3x3) + ReLU + max pool | 16 x 32 x 32 |
| Conv (16 to 32 channels, 3x3) + ReLU + max pool | 32 x 16 x 16 |
| Flatten | 8,192 |
| Linear | 10 (one score per class) |

Pixel values are divided by 255 so they fall between 0 and 1. The network was trained with cross-entropy loss and the Adam optimizer (learning rate 0.001), in batches of 64 for 20 epochs, on an RTX 3060 GPU. After each epoch it was scored on the validation set, and the weights from the epoch with the highest validation accuracy were saved as `best_model.pt`.

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

All models were scored on the same 5,400 held-out test images.

| Model | Test accuracy |
|---|---|
| Random guessing | about 10% |
| Linear baseline (SGDClassifier, raw pixels) | 35.7% |
| CNN (PyTorch, 20 epochs, best epoch by validation) | 81.6% |

### Baseline

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

### Baseline caveats

- This is a single run (`random_state=42`), with no repeats, so small differences between classes may just be noise.
- Pixels were used unscaled and the classifier has its default settings.
- Training took about 9 minutes on the CPU.
- A first attempt with `LinearSVC` on the full training set ran for over an hour without finishing and was stopped, so `SGDClassifier` was used instead.

### CNN

The CNN scored 81.6% accuracy on the 5,400 test images, against 35.7% for the baseline on the same images. The model was selected on the validation set, where it scored 82.0% (epoch 18 of 20); the test and validation figures are similar, which suggests the epoch choice did not overfit to the validation set. The test set was scored once, after training was finished. Training took about 2 seconds per epoch on the GPU.

In the first training run, validation loss was lowest (0.527) at epoch 15, and training loss reached 0.292 by epoch 20. The gap between training and validation loss widened, an early sign of overfitting.

### CNN caveats

- This is a single training run with no random seed, so every run gives slightly different results.
- The best epoch was chosen on validation accuracy, which was noisy: it moved between about 0.757 and 0.820 over the last 8 epochs, so the choice of epoch is partly down to chance.
- The comparison with the baseline is not a controlled test of CNN versus non-CNN. The baseline was a linear model on raw pixels with default settings, so the result shows the CNN beats that simple floor, not that CNNs beat classical methods in general.
- `best_model.pt` is not in the repository (`*.pt` files are ignored). Running `train_cnn.py` again produces different weights and overwrites the file, so the 81.6% would no longer describe it.
- Per-class results for the CNN have not been analysed yet.

## Citation
Helber, P., Bischke, B., Dengel, A., & Borth, D. (2018). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. Zenodo. https://doi.org/10.5281/zenodo.7711810