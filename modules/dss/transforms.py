from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    AsDiscrete,
    ScaleIntensityd,
    EnsureTyped,
)
from .config import N_CLASSES

load_transforms = Compose([
    LoadImaged(keys=["img"]),
    AddChanneld(keys=["img"]),
    EnsureTyped(keys=["img"]),
])

pre_transforms = Compose([
    EnsureTyped(keys=["img"]),
    ScaleIntensityd(keys=["img"]),
    EnsureTyped(keys=["img"]),
])

post_transforms = Compose([
    EnsureTyped(keys=["img"]),
    AsDiscrete(argmax=True, to_onehot=N_CLASSES)
])