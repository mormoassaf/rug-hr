import os
import random as rnd
from PIL import Image, ImageFont, ImageDraw
import cv2
import skimage
import re
import numpy as np
from ..utils.transforms import crop_foreground, match_prototype

PAPER_MARGIN = 32


def time2token(text_corpus, time):
    # Split words, special characters, and spaces into separate tokens
    tokens = re.findall(r"[\w']+|[.,!?;:\-()\[\]{}\"\\]|[\s]", text_corpus)
    # If 'time' is within the range of the length of tokens, return the token at the given index
    if 0 <= time < len(tokens):
        return tokens[time]
    else:
        return None


def token2image(token, font_size, font_path=None, loaded_font=None, crop=True):
    if font_path is None and loaded_font is None:
        raise ValueError("Either font_path or font must be provided.")
    if font_path is not None and loaded_font is not None:
        raise ValueError("Only one of font_path or font must be provided.")
    if font_path is not None:
        if not os.path.exists(font_path):
            raise ValueError("font_path does not exist.")
        loaded_font = ImageFont.truetype(font_path, font_size)
    assert loaded_font is not None
    image = Image.new("RGB", (1024, 256), (255, 255, 255))  # Creating a white image
    draw = ImageDraw.Draw(image)
    draw.text(
        (PAPER_MARGIN, PAPER_MARGIN),
        token,
        font=loaded_font,
        fill=(0, 0, 0)
    )  # Drawing the token text on the image
    # crop to the height of the first letter
    if crop:
        token_height = loaded_font.getsize(token)[1]
        image = image.crop((0, 0, image.width, token_height))

    image = np.array(image)  # Converting the image to a numpy array
    image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    image = image / 255.0  # Normalizing the image
    image = 1 - image  # Inverting the image
    image = (image > 0.5).astype(np.float32)  # Binarizing the image

    return image


def label_chars_in_token(token, *args, **kwargs):
    token_image = token2image(token, crop=False, *args, **kwargs)
    acc = np.zeros(token_image.shape)

    (i_h, i_w) = token_image.shape
    x = 0
    max_deviation = 0

    for (i, c) in enumerate(token):
        kernel = token2image(c, crop=False, *args, **kwargs)  # (w, font_size)
        kernel = crop_foreground(kernel)

        (k_h, k_w) = kernel.shape
        _, locations = match_prototype(
            image=token_image,
            template=kernel)
        i_pos, j_pos = locations
        # get pos that is closest to x from the right
        j_distances = j_pos - x
        j_distances[j_distances < -max_deviation] = np.iinfo(np.int32).max

        location = (i_pos[np.argmin(j_distances)], j_pos[np.argmin(j_distances)])

        # ascii label
        label = ord(c) - 32
        s_i = max(min(location[0], i_h - k_h), 0)
        s_j = max(min(location[1], i_w - k_w), 0)

        # kernel *= label
        overlap = np.where(acc[s_i:s_i + k_h, s_j:s_j + k_w] * kernel > 0, 1, 0)
        kernel -= overlap
        acc[s_i:s_i + k_h, s_j:s_j + k_w] += kernel * label

        x = s_j + k_w
        max_deviation = k_w // 2

    return acc


def generate_image_from_token(token, font_size, loaded_font=None, font_path=None):
    if loaded_font is None and font_path is None:
        raise ValueError("Either loaded_font or font_path must be provided.")
    if loaded_font is not None and font_path is not None:
        raise ValueError("Only one of loaded_font or font_path must be provided.")
    if font_path is not None:
        if not os.path.exists(font_path):
            raise ValueError("font_path does not exist.")
        loaded_font = ImageFont.truetype(font_path, 64)
    assert loaded_font is not None

    img = label_chars_in_token(token, loaded_font=loaded_font, font_size=font_size)
    img = img.astype(np.uint8)
    img = crop_foreground(img, axis=0)  # crop the sides

    font_height = loaded_font.getsize(token[0])[1]
    img = img[PAPER_MARGIN:font_height + 2 * PAPER_MARGIN, :]

    # aspect_ratio = img.shape[1] / img.shape[0]
    # new_h = font_size
    # new_w = int(aspect_ratio * new_h)
    # img = skimage.transform.resize(img, (new_h, new_w))
    return img
