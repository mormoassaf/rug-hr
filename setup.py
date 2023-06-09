# Purpose: Setup file for the package
import os

ARTIFACT_REPISOTRY = [
    {
        "name": "seglm-v1-256x256-dss.pt",
        "url": "https://drive.google.com/u/1/uc?id=15tu0ucEtM2anjKGHBJt70_ro7HUVHxS_&export=download",
    }
]

ARTIFACTS_CACHE = "./artifacts"

def download_artifacts():
    if not os.path.exists(ARTIFACTS_CACHE):
        os.makedirs(ARTIFACTS_CACHE)

    for artifact in ARTIFACT_REPISOTRY:
        artifact_path = os.path.join(ARTIFACTS_CACHE, artifact["name"])
        if os.path.exists(artifact_path):
            continue

        print(f"Downloading {artifact['name']}...")
        os.system(f"wget -O {artifact_path} {artifact['url']}")


def setup_conda():
    # Check if the environment has already been installed, if so then return
    check_list = os.system("conda env list")
    if "handrec" in check_list:
        return
    os.system("conda env create -f environment.yml")
    os.system("conda activate handrec")

def setup():
    download_artifacts()
    setup_conda()

if __name__ == "__main__":
    setup()     