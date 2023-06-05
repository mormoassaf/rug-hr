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
    ScaleIntensityRanged,
)
import torch
import random
from skimage.transform import resize


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

  
class AddMask(Transform):
    """ Adds a mask channel to the image where 
    Args: 
        prob: probability of zeroing out a region
        min_portion: minimum portion of the image to zero out in each dimension
        max_portion: maximum portion of the image to zero out in each dimension
    """
    def __init__(self, 
                 prob=0.5, 
                 min_portion=(0.1, 0.1), 
                 max_portion=(0.5, 0.5)):
        self.prob = prob
        self.min_portion = min_portion
        self.max_portion = max_portion
    
    """
    Args:
        image: (..., C, H, W)
    Returns: with mask channel added
        image: (..., C+1, H, W)
    """
    def __call__(self, image):
        C, H, W = image.shape[-3:]
        mask = torch.ones((*image.shape[:-3], 1, H, W))
        if random.random() < self.prob:
            min_p_i, min_p_j = self.min_portion
            max_p_i, max_p_j = self.max_portion
            p_i = random.uniform(min_p_i, max_p_i)
            p_j = random.uniform(min_p_j, max_p_j)
            size_i = int(H * p_i)
            size_j = int(W * p_j)
            i_start = random.randint(0, H - size_i)
            j_start = random.randint(0, W - size_j)
            i_end = i_start + size_i
            j_end = j_start + size_j   
            mask[..., i_start:i_end, j_start:j_end] = 0
        image *= mask
        return torch.cat([image, mask], dim=-3)
    
class AddMaskd(MapTransform):
    def __init__(self, keys, allow_missing_keys: bool = False,  *argc, **argv) -> None:
        super().__init__(keys, allow_missing_keys)
        self.add_mask = AddMask(*argc, **argv)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.add_mask(d[key])
        return d
    
# Transofrmation that resizes teh axis to a number while keeping the aspect ratio
class ResizeAxis(Transform):
    def __init__(self, axis, size, scale_axes=tuple(range(1, 4))):
        self.axis = axis
        self.size = size
        self.scale_axes = scale_axes

    def __call__(self, data):
        shape = data.shape
        scale = self.size / shape[self.axis]
        new_shape = list(shape)
        for i in self.scale_axes:
            if i == self.axis:
                new_shape[i] = self.size
            else:
                new_shape[i] = int(shape[i] * scale)
        data = resize(data, new_shape, anti_aliasing=False)
        return torch.tensor(data, dtype=torch.float32)

class ResizeAxisd(MapTransform):
    def __init__(self, keys, axis, size, scale_axes=tuple(range(1, 4))):
        super().__init__(keys)
        self.resize_axis = ResizeAxis(axis, size, scale_axes)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.resize_axis(d[key])
        return d


load_transforms = Compose([
    LoadImaged(keys=["img"]),
    EnsureChannelFirstd(keys=["img"]),
    EnsureTyped(keys=["img"]),
])

pre_transforms = Compose([
    Transposed(keys=["img"], indices=[0, -1, -2]),
    ScaleIntensityd(keys=["img"]),
    Binarized(keys=["img"]),
    CropForegroundd(keys=["img"], source_key="img"),
    Invertd(keys=["img"]),
    ResizeAxisd(keys=["img"], axis=-2, size=100, scale_axes=[-1, -2]),
    AddMaskd(
        keys=["img"], 
        prob=0.0, 
        min_portion=(0.8, 0.01), 
        max_portion=(0.9, 0.05),
    ),
    # MaxPoold(keys=["img"], kernel_size=2, stride=1, padding=1),
    EnsureTyped(keys=["img"]),
])