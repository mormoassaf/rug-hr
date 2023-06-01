from monai.transforms import (
    Compose,
    LoadImaged,
    Resized,
    ScaleIntensityd,
    EnsureTyped,
    EnsureChannelFirstd,
    Transposed,
    MapTransform,
    Transform,
    ThresholdIntensityd,
    CropForegroundd,
    GaussianSmoothd,
)
import torch

class Invert(Transform):
    def __call__(self, data, max_value=1.0, min_value=0.0):
        return max_value - data + min_value

class Invertd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        self.invert = Invert()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.invert(d[key])
        return d

class Binarize(Transform):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, data):
        threshold = self.threshold
        data[data >= threshold] = 1.0
        data[data < threshold] = 0.0
        return data
    
class Binarized(MapTransform):
    def __init__(self, keys, *args, **kwargs):
        super().__init__(keys)
        self.binarize = Binarize(*args, **kwargs)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.binarize(d[key])
        return d
    
class MaxPool(Transform):
    def __init__(self, *args, **kwargs):
        self.pool = torch.nn.MaxPool2d(*args, **kwargs)

    def __call__(self, data):
        return self.pool(data)

class MaxPoold(MapTransform):
    def __init__(self, keys, *args, **kwargs):
        super().__init__(keys)
        self.pool = MaxPool(*args, **kwargs)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.pool(d[key])
        return d

load_transforms = Compose([
    LoadImaged(keys=["img"]),
    EnsureChannelFirstd(keys=["img"]),
    EnsureTyped(keys=["img"]),
])

pre_transforms = Compose([
    EnsureTyped(keys=["img"]),
    GaussianSmoothd(keys=["img"], sigma=0.2),
    Binarized(keys=["img"], threshold=0.2),
    Invertd(keys=["img"]),
    MaxPoold(keys=["img"], kernel_size=4, stride=2, padding=1),
    # CropForegroundd(keys=["img"], source_key="img", select_fn=lambda x: x > 0, margin=0),
    # ScaleIntensityd(keys=["img"]),
    # Resized(keys=["img"], spatial_size=(1512, 1024)),
    # scale down
    EnsureTyped(keys=["img"]),
])