
import torch
import glob

from modules.iam import SegLMIAM
from modules.iam.postprocessing import transcribe_image as transcribe_image_iam
from modules.dss import SegLMDSS
from modules.dss.postprocessing import transcribe_image as transcribe_image_dss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("WARNING: CUDA is not available, using CPU instead.")
    confirmation = input("Do you want to continue? CPU computation will take very long (y/n): ")

INPUT_PATTERN = input("Enter the path pattern to the binarized images (e.g., ./images/*binarized*): ")
OUTPUT_PATH = input("Enter the path to the output folders (e.g., ./outputs): ")
MODEL = input("Enter the model to use (e.g., iam or dss): ")

model = None
if MODEL == "iam":
    model = SegLMIAM("./artifacts/seglm-v1-256x256-iam.pt")
elif MODEL == "dss":
    model = SegLMDSS("./artifacts/seglm-v1-256x256-dss.pt")