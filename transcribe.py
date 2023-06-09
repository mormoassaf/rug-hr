
import torch
from glob import glob
from tqdm import tqdm
import os

from modules.iam import SegLMIAM
from modules.iam.postprocessing import transcribe_image as transcribe_image_iam
from modules.dss import SegLMDSS
from modules.dss.postprocessing import transcribe_image as transcribe_image_dss

print("HandRec - Handwritten Text Recognition")
print("======================================")
print("This program will transcribe a set of images using a pre-trained model.")
print("DSS: Dead Sea Scrolls")
print("IAM: IAM Handwriting Database")
print("======================================")


if not torch.cuda.is_available():
    print("WARNING: CUDA was not detected, using CPU instead.")
    confirmation = input("Do you want to continue? CPU computation will take very long (y/n): ")
    if confirmation != "y":
        print("Aborting...")
        exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_PATTERN = input("Enter the path pattern to the input images (e.g., ./images/*binarized*): ")
OUTPUT_PATH = input("Enter the path to the output folder (default: ./outputs): ")
if OUTPUT_PATH == "":
    OUTPUT_PATH = "./outputs"

MODEL = input("Enter the model to use (e.g., iam or dss): ")

model = None
if MODEL == "dss":
    model = SegLMDSS().load("./artifacts/seglm-v1-256x256-dss.pt").to(DEVICE)
elif MODEL == "iam":
    model = SegLMIAM().load("./artifacts/seglm-masked-v1-128x1024-iam.pt").to(DEVICE)

if MODEL not in ["iam", "dss"]:
    print("Invalid model, aborting...")
    exit(1)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

transcribe = transcribe_image_dss if MODEL == "dss" else transcribe_image_iam

def main():
    print(f"Using device: {DEVICE}")
    print(f"Input pattern: {INPUT_PATTERN}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Model: {MODEL}")

    for im_path in tqdm(glob(INPUT_PATTERN)):

        # Predict the segmentation and transcribe the image
        segmentation = model.predict(im_path=im_path)
        transcription = transcribe(segmentation)
        
        # Write the transcription to a file
        im_name = str(os.path.basename(im_path))
        im_name = im_name[:im_name.rfind(".")]
        out_path = os.path.join(OUTPUT_PATH, f"{im_name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(transcription)

if __name__ == "__main__":
    main()