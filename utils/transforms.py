import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ensure is in parent directory
try:
    os.chdir("../../RUG-HandRec/")
except:
    pass


def walk_char(acc, i, j, visited_symbols=[], visited_acc=None):
    stack = [(i, j)]
    while stack:
        i, j = stack.pop()
        # check if i and j are within bounds
        if i < 0 or i >= acc.shape[0] or j < 0 or j >= acc.shape[1]:
            continue

        # if visited or reached a background pixel then return
        if acc[i, j] == 0 or visited_acc[i, j] == 1:
            continue

        visited_acc[i, j] = 1  # visited
        visited_symbols.append((i, j))

        stack.append((i - 1, j))
        stack.append((i + 1, j))
        stack.append((i, j - 1))
        stack.append((i, j + 1))


def set_adaptive_threshold(acc):
    num_pixels_per_symbol = []
    visited_acc = np.zeros_like(acc)
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            if acc[i, j] == 1 and visited_acc[i, j] == 0:
                visited_positions = []
                walk_char(acc, i, j, visited_positions, visited_acc=visited_acc)
                num_pixels_per_symbol.append(len(visited_positions))

    num_pixels_per_symbol = np.sort(num_pixels_per_symbol)
    print(num_pixels_per_symbol)

    fourier = np.fft.fft(num_pixels_per_symbol)
    freqs = np.fft.fftfreq(len(num_pixels_per_symbol))
    fourier[freqs > 0.1] = 0
    num_pixels_per_symbol = np.fft.ifft(fourier)
    num_pixels_per_symbol = np.real(num_pixels_per_symbol)
    num_pixels_per_symbol[num_pixels_per_symbol < 0] = 0
    num_pixels_per_symbol = np.round(num_pixels_per_symbol).astype(np.uint8)

    threshold = num_pixels_per_symbol[-1]

    return threshold


def remove_ornaments(page, threshold=10):
    acc = page.copy()
    visited_acc = np.zeros_like(acc)
    delete_list = []

    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            if acc[i, j] == 1 and visited_acc[i, j] == 0:

                visited_positions = []
                walk_char(acc, i, j, visited_positions, visited_acc=visited_acc)

                if len(visited_positions) < threshold:
                    delete_list.extend(visited_positions)

            visited_acc[i, j] = 1  # visited

    for i, j in delete_list:
        acc[i, j] = 0
    acc = (acc > 0)
    acc = acc.astype(np.uint8)
    return acc


if __name__ == '__main__':
    masks_folder = "experiments/Masks/"
    masks = os.listdir(masks_folder)
    random_mask = np.random.choice(masks)
    image = np.array(Image.open(masks_folder + random_mask))

    image_binary = np.where(image > 0, 1, 0)
    plt.imshow(image_binary)
    plt.show()

    image_without_ornaments = remove_ornaments(
        image_binary,
        threshold=set_adaptive_threshold(image_binary)
    )

    plt.imshow(image_without_ornaments)
    plt.show()
