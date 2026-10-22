from pathlib import Path
import numpy as np
from load_data import load_images

images, labels = load_images(Path("data/EuroSAT_RGB"))
np.save("data/images.npy", images)
np.save("data/labels.npy", labels)