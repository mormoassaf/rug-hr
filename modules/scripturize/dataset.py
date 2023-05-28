import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from .generate import sample_manuscript_from_corpus


def sample_text_from_corpus(text_corpus, max_length=8192):
    if len(text_corpus) - max_length < 0:
        return text_corpus
    page = np.random.randint(0, len(text_corpus) - max_length)
    text = text_corpus[page:page+max_length]
    return text


def generate_dataset_sample(i, text_corpus, out_dir, file_alias="manu", save=True, *argc, **argv):
    text = sample_text_from_corpus(text_corpus)
    manuscript = None
    while manuscript is None:
        try:
            manuscript, params = sample_manuscript_from_corpus(text, *argc, **argv)
        except Exception as e:
            print(e)
            manuscript = None
    raw_manuscript = manuscript["raw"].astype(np.uint8)
    mutated_manuscript = manuscript["mutated"].astype(np.uint8)

    # Save dataset
    raw_manuscript = Image.fromarray(raw_manuscript)
    mutated_manuscript = Image.fromarray(mutated_manuscript)

    if save:
        raw_manuscript.save(os.path.join(out_dir, "Raw", f"{file_alias}_{i}.png"))
        mutated_manuscript.save(os.path.join(out_dir, "Mutated", f"{file_alias}_{i}.png"))

    return (manuscript, text, params)


def generate_manuscript_dataset(
        text_corpus: str, 
        dataset_size=1024, 
        out_dir="./datasets/generated_manuscripts", 
        save=True, 
        parallel=True,
        *argc, **argv):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.makedirs(os.path.join(out_dir, "Raw"))
        os.makedirs(os.path.join(out_dir, "Mutated"))
        
    print("SCRIPTURIZE GENERATOR :: generating dataset...")
    if parallel:
        with ThreadPoolExecutor() as executor:
            dataset = list(
                tqdm(
                    executor.map(
                        lambda i: generate_dataset_sample(i, text_corpus, out_dir, save=save, *argc, **argv),
                        range(dataset_size)
                    ),
                    total=dataset_size,
                )
            )
    else:
        dataset = []
        for i in tqdm(range(dataset_size)):
            dataset.append(generate_dataset_sample(i, text_corpus, out_dir, save=save, *argc, **argv))

    return dataset
