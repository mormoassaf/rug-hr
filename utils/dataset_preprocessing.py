import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from skimage.transform import resize
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

PATH_TO_DATASET = Path('./datasets/monkbrill2')
PATH_TO_PROCESSED_DATASET = Path('./datasets/monkbrill2_processed')

character_thresholds = {
    "Alef": 40,
    "Bet": 30,
    "Gimel": 50,
    "Dalet": 20,
    "He": 20,
    "Waw": 20,
    "Zayin": 20,
    "Het": 30,
    "Tet": 20,
    "Yod": 30,
    "Kaf": 80,
    "Lamed": 30,
    "Mem": 30,
    "Nun-medial": 150,
    "Samekh": 50,
    "Ayin": 50,
    "Pe": 20,
    "Tsadi-medial": 120,
    "Qof": 70,
    "Resh": 50,
    "Shin": 20,
    "Taw": 40,
    "Kaf-final": 30,
    "Mem-medial": 30,
    "Nun-final": 30,
    "Pe-final": 40,
    "Tsadi-final": 30
}

char2token = {
    "א": "Alef",
    "ב": "Bet",
    "ג": "Gimel",
    "ד": "Dalet",
    "ה": "He",
    "ו": "Waw",
    "ז": "Zayin",
    "ח": "Het",
    "ט": "Tet",
    "י": "Yod",
    "כ": "Kaf",
    "ל": "Lamed",
    "מ": "Mem",
    "נ": "Nun-medial",
    "ס": "Samekh",
    "ע": "Ayin",
    "פ": "Pe",
    "צ": "Tsadi-medial",
    "ק": "Qof",
    "ר": "Resh",
    "ש": "Shin",
    "ת": "Taw",
    "ך": "Kaf-final",
    "ם": "Mem-medial",
    "ן": "Nun-final",
    "ף": "Pe-final",
    "ץ": "Tsadi-final",
}


def jpg_to_binarized_array(jpg_file, threshold=0.5):
    # Read the JPG file as a PIL Image
    image = Image.open(jpg_file).convert('L')  # Convert to grayscale

    # Convert the PIL Image to a numpy array
    image_np = np.array(image)

    # Binarize the image using thresholding
    binarized_image = (image_np > threshold * 255).astype(np.uint8)

    return binarized_image


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


def remove_ornaments(page, threshold=20):
    page = 1 - page

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


def make_uniform(binarized_array, height):
    # Find the bounding box around the black pixels
    binarized_array = 1 - binarized_array
    minr, minc, maxr, maxc = regionprops(label(binarized_array))[0].bbox

    # Extract the character
    character = binarized_array[minr:maxr, minc:maxc]

    # Calculate the new height and width while keeping the aspect ratio
    new_height = height
    aspect_ratio = character.shape[1] / character.shape[0]
    new_width = int(new_height * aspect_ratio)

    # Resize the character
    resized_character = resize(character, (new_height, new_width), order=0, preserve_range=True,
                               anti_aliasing=False).astype(
        np.uint8)

    return 1 - resized_character


def process_image(jpg_file, threshold, height=64):
    binarized_array = jpg_to_binarized_array(jpg_file)

    # remove ornaments
    binarized_array = remove_ornaments(binarized_array, threshold=threshold)
    binarized_array = 1 - binarized_array

    # create a uniform image
    uniform_image = make_uniform(binarized_array, height=height)

    return uniform_image


def process_dataset(thresholds_dict):
    input_folder = PATH_TO_DATASET
    output_folder = PATH_TO_PROCESSED_DATASET

    for letter_folder in tqdm(input_folder.iterdir()):
        if letter_folder.is_dir():
            letter = letter_folder.name

            # Create the corresponding output folder for the letter
            letter_output_folder = output_folder / letter
            letter_output_folder.mkdir(parents=True, exist_ok=True)

            threshold = thresholds_dict[letter]

            # Iterate over all images inside the letter folder
            for image_path in letter_folder.glob('*.jpg'):
                output_path = letter_output_folder / image_path.name

                # Process and save the image
                processed_image = process_image(image_path, threshold)
                plt.imsave(output_path, processed_image, cmap='gray')


def cache_char_dataset():
    cache = {}

    for letter_folder in tqdm(PATH_TO_PROCESSED_DATASET.iterdir()):
        if letter_folder.is_dir():
            if letter_folder.name not in cache:
                cache[letter_folder.name] = []
            for image_path in letter_folder.glob('*.jpg'):
                image = plt.imread(image_path)
                cache[letter_folder.name].append(image)

    return cache


def sample_char(char: str, height: int, cache=None) -> np.ndarray:
    folder_name = char2token[char]
    if cache:
        images = cache[folder_name]
        choice = np.random.choice(len(images))
        image = images[choice].copy()
    else:
        folder_path = PATH_TO_PROCESSED_DATASET / folder_name
        random_image_path = np.random.choice(list(folder_path.glob('*.jpg')))
        image = plt.imread(random_image_path)
    return image


if __name__ == '__main__':
    # process_dataset(character_thresholds)  # Run only once

    char = np.random.choice(list(char2token.keys()))
    # char = 'צ'
    image = sample_char(char, height=64)
    plt.imshow(image, cmap='gray')
    plt.show()
