# Handwriting Recognition 2023

## Requirements

- Anaconda must be installed.
- You must have a CUDA-enabled device avialable to carry out the inference

## Installation
Run the following command
```bash
python setup.py
```
This will download the model artifacts and setup the conda environment `handrec`. The models are downloaded from filesender.surf, and the links expires on 30/06/2023.

**PLEASE NOTE**: 
- The model artifacts are large and may take a while to download.
- We only accept binarized images as input for the dead sea scrolls. Please binarize your images before running the inference.


### Alternative Installation
You can create your own environment with conda or pip. 
```
pip install -r requirements.txt
```
Make sure torch can detect your CUDA device.
The download links for the models are in the setup.py file. You can download them manually and place them in the `artifacts` folder with their respective names.

Models can also be obtained on Google drive:
- `seglm-v1-256x256-dss.pt`: https://drive.google.com/u/1/uc?id=15tu0ucEtM2anjKGHBJt70_ro7HUVHxS_&export=download
- `seglm-masked-v1-128x1024-iam.pt`: https://drive.google.com/u/1/uc?id=1vWecIuCPiPKSJO1I1hkS5a3SySvLKnyq&export=download


## Usage
```bash
conda activate handrec
```

To make the transcriptions run our interactive script:
```bash
python transcribe.py
```

If the script fails you may interact with our notebooks in:
- [iam_eval.ipynb](./experiments/iam_eval.ipynb)
- [dss_eval.ipynb](./experiments/dss_eval.ipynb)
