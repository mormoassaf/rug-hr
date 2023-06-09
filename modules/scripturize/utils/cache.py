import os
from PIL import Image
import random
import numpy as np


class DictCache:

    def __init__(self, data: dict):
        self.data = data

    def sample(self):
        choice = random.choice(list(self.data.keys()))
        return self.data[choice]


"""Load images from a directory into a DictCache object. 
Args:
    dir_path (str): Path to the directory containing the images.
Returns:
    DictCache: A DictCache object containing the images.
"""


def cache_images(dir_path: str) -> DictCache:
    images = {}
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        image = Image.open(filepath)
        images[filename] = np.array(image).astype(np.float32)
    print("Loaded images from directory: " + dir_path + ".")
    return DictCache(images)
