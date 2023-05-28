import os
import numpy as np
from PIL import Image
from skimage import transform
from .utils.cache import cache_images

WARNINGS_ENABLED = True

"""Generates images of a manuscript from a text corpus with variance in writing style.

Args:
    text_corpus (any): data object used by time2token to generate tokens
    time2token (function): (text_corpus, time) -> token
    token2image (function,): (token, font_size) -> image with 0 for background and positive values for foreground
    spacing_token (str): token used to represent spacing between words
    font_size (int): font size of the manuscript
    size (tuple): size of the manuscript in pixels
    margin (int): margin of the manuscript in pixels
    char_spacing (int): average spacing between characters in pixels
    word_spacing (int): average spacing between words in pixels
    char_spacing_variance (float): determines max deviation from average spacing between characters
    word_spacing_variance (float): determines max deviation from average spacing between words
    line_spacing (int): average spacing between lines in pixels
    char_line_shift (int): average shift of characters from the line in pixels
    allow_intersections (bool): whether characters are allowed to touch
    p_corrupt (float): probability of the manuscript to sample a corruption prototype
    p_prototype (float): probability of the manuscript to sample a scripture prototype
    direction (int): direction of writing, 1 for left-to-right, -1 for right-to-left
    scripture_prototypes_path (str): path to the directory containing scripture prototypes
    scripture_corruption_prototypes_path (str): path to the directory containing scripture corruption prototypes
    on_token_paste (function): (dict) -> None, called when a token is pasted onto the manuscript

Returns: a dictionary containing the manuscript image and the tokens used to generate it with the following keys:
    "raw" (np.ndarray): the manuscript image without prototypes
    "prototype" (np.ndarray): the manuscript that was used in the generation process
    "corruption_prototype" (np.ndarray): the corruption prototype that was used in the generation process
    "mutated": (np.ndarray): the manuscript image with prototypes
"""
def generate_manuscript_from_corpus(
        text_corpus, 
        time2token,
        token2image,
        spacing_token=" ",
        font_size=32, 
        size=(1024, 512), 
        margin=64, 
        char_spacing=20,
        word_spacing=32,
        char_spacing_variance=0.2,
        word_spacing_variance=0.4,
        line_spacing=2,
        char_line_shift=2,
        allow_intersections=True,
        p_corrupt=0.2,
        p_prototype=0.2,
        direction=1,
        scripture_prototypes_cache=cache_images("./assets/scripture_prototypes/"),
        scripture_corruption_prototypes_cache=cache_images("./assets/scripture_corruption_prototypes/"),
        on_token_paste=None,
    ):
    
    ## INITIALIZATION SETTINGS

    height, width = size
    manuscript = np.zeros((size[0], size[1]), dtype=np.uint8)
    page_bounds = (margin, margin, height-margin, width-margin) # (i_min, j_min, i_max, j_max)
    
    offset = (font_size, font_size)
    init_pos = (margin + offset[0], margin + offset[1])
    if direction == -1:
        init_pos = (init_pos[0] - offset[0], page_bounds[3] - offset[1])
    line_phase = np.random.rand() * width
    frequency_phase_factor = np.random.rand()*0.4 + 0.8

    ## VARIABLES
    t = 0
    positions = [init_pos]

    ## SAMPLERS

    def next_char():
        nonlocal text_corpus, t
        t += 1
        return time2token(text_corpus, t)

    def sample_char_spacing():
        upper_bound = int(char_spacing*(1+char_spacing_variance))
        lower_bound = int(char_spacing*(1-char_spacing_variance))
        if allow_intersections:
            lower_bound = 0
        return np.random.randint(lower_bound, upper_bound)
    
    def sample_word_spacing():
        return np.random.randint(int(word_spacing*(1-word_spacing_variance)), int(word_spacing*(1+word_spacing_variance)))
    
    def sample_char_line_shift(x):
        span = width - 2*margin
        omega = np.pi / span
        shift_factor = np.sin(x * omega * frequency_phase_factor - line_phase)
        shift = char_line_shift*shift_factor
        var_factor = 1 + char_spacing_variance*np.random.rand()
        shift *= var_factor
        return int(shift)
    
    def sample_prototype(cache=scripture_prototypes_cache):
        prototype = cache.sample()
        if len(prototype.shape) == 3:
            # conver to grayscale
            prototype = np.dot(prototype[...,:3], [0.299, 0.587, 0.114])
        # preprocess
        prototype = transform.resize(prototype, (height, width))
        # random flip around x or y axis
        if np.random.rand() > 0.5:
            prototype = np.flip(prototype, axis=0)
        if np.random.rand() > 0.5:
            prototype = np.flip(prototype, axis=1)
        # augmentation
        angle = np.random.randint(-10, 10)
        prototype = transform.rotate(prototype, angle, resize=False, center=None, order=1, mode='constant', cval=0, clip=True, preserve_range=False)
        # normalise
        prototype = prototype / 255
        prototype = (prototype > 0.5).astype(np.uint8)
        prototype = prototype.astype(np.uint8)
        return prototype    
    
    def sample_char_img(char):
        img = token2image(char, font_size)
        if (img is None):
            if WARNINGS_ENABLED:
                print(f"Warning: token2image({char}) return None. Proceeding to next character.")
            while img == None:
                char = next_char()
                img = token2image(char, font_size)
        return img    

    # sliding window to generate the manuscript
    while positions:

        i, j = positions.pop(0)

        # check if the page is full
        if (i+font_size >= page_bounds[2]):
            break

        char = next_char()
        if char == None:
            break
        char_space = sample_char_spacing()
        line_shift = sample_char_line_shift(x=j)

        if (char == spacing_token):
            positions.append((i, j + direction*sample_word_spacing()))
            continue

        # generate the image of the token
        img = sample_char_img(char)
        
        # paste the image on the manuscript
        font_height, font_width = img.shape
        i_paste_range = (i + line_shift, i + line_shift+font_height)
        j_paste_range = (j, j + direction*font_width)
        j_paste_range = (min(j_paste_range), max(j_paste_range))
        assert (i_paste_range[1] - i_paste_range[0] == img.shape[0])
        assert (j_paste_range[1] - j_paste_range[0] == img.shape[1])

        # Check if we need to proceed to the next line by checking if the character is out of bounds
        if (
            (i_paste_range[0] < page_bounds[0]) or
            (i_paste_range[1] >= page_bounds[2]) or
            (j_paste_range[0] < page_bounds[1]) or
            (j_paste_range[1] >= page_bounds[3]) 
            ):
            next_i = i + font_size+char_space+line_spacing
            next_j = int(init_pos[1] + direction*np.random.rand()*word_spacing)
            positions.append((next_i, next_j))
            continue

        # Update the manuscript by pasting the image into the bounding box
        overlap = np.where(manuscript[i_paste_range[0]:i_paste_range[1], j_paste_range[0]:j_paste_range[1]] * img, 1, 0).astype(np.uint8)
        # zero out overlap
        img *= (1 - overlap)
        manuscript[i_paste_range[0]:i_paste_range[1], j_paste_range[0]:j_paste_range[1]] += img.astype(np.uint8)
        # Proceed to the next character on the same line
        positions.append((i, j + direction*(font_width+char_space)))

        ## EVENTS
        if on_token_paste:
            on_token_paste({"token": char, "position": (i, j), "font_size": font_size, "char_spacing": char_space, "line_shift": line_shift, "line_spacing": line_spacing})
    
    ## ADD CORRUPTIONS AND PROTOTYPES
    mutated_manuscript = manuscript.copy()
    prototype = None
    corruption_proto = None
    if np.random.rand() < p_prototype:
        prototype = sample_prototype()
        mutated_manuscript *= prototype 
    if np.random.rand() < p_corrupt:
        corruption_proto = sample_prototype(cache=scripture_corruption_prototypes_cache)
        mutated_manuscript *= corruption_proto
    
    return {
        "raw": manuscript,
        "prototype": prototype,
        "corruption_prototype": corruption_proto,
        "mutated": mutated_manuscript
    }

"""Uses the generate_manuscript function to generate a manuscript from a corpus of text.
Args:
    corpus (any): the text object being sampled from
    font_size_range (tuple): the range of font sizes to sample from
    char_spacing_range (tuple): the range of character spacings to sample from
    char_line_shift_range (tuple): the range of character line shifts to sample from
    line_spacing_range (tuple): the range of line spacings to sample from
    *argc: additional arguments to pass to generate_manuscript
    **argv: additional keyword arguments to pass to generate_manuscript

Returns:
    (dict: a dictionary containing the raw manuscript, the prototype, the corruption prototype, and the mutated manuscript.,
    dict: a dictionary containing the parameters used to generate the manuscript)
"""
def sample_manuscript_from_corpus(corpus,
        font_size_range=(32, 64),
        char_spacing_range=(10, 40),
        char_line_shift_range=(5, 10),
        line_spacing_range=(20, 64),
        *argc, **argv):


    font_size = np.random.randint(*font_size_range)
    char_spacing = np.random.randint(*char_spacing_range)
    char_line_shift = np.random.randint(*char_line_shift_range)
    word_spacing = int(char_spacing * (4.0*np.random.rand() + 4.0))
    line_spacing = np.random.randint(*line_spacing_range)

    # Create the manuscript
    generated_objects = generate_manuscript_from_corpus(
        corpus, 
        font_size=font_size, 
        char_spacing=char_spacing,
        word_spacing=word_spacing,
        char_line_shift=char_line_shift,
        line_spacing=line_spacing,
        *argc, **argv
    )

    return generated_objects, {
        "font_size": font_size,
        "char_spacing": char_spacing,
        "word_spacing": word_spacing,
        "char_line_shift": char_line_shift,
        "line_spacing": line_spacing,
    }

