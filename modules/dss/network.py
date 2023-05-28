
from monai.networks.nets import UNETR
from .config import SPATIAL_SIZE, N_CLASSES
import torch
import numpy as np
import os
from .transforms import load_transforms, pre_transforms, post_transforms
from monai.inferers import SlidingWindowInferer

class SegFormerDSS(UNETR):
    def __init__(self, **kwargs):
        super().__init__(
            in_channels=1,
            out_channels=N_CLASSES,
            img_size=SPATIAL_SIZE,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.3,
            spatial_dims=2,
            **kwargs
        )
        self.inferer = SlidingWindowInferer(roi_size=SPATIAL_SIZE, sw_batch_size=16, overlap=0.8)

    def load(self, path):
        state = torch.load(os.path.join(os.getcwd(), path), map_location=torch.device('cpu'))
        self.load_state_dict(state)
        return self

    def predict(self, images=None, im_path=None, return_tensors="np", batch_size=16):
        """
        Args:
            images (numpy.array): input data of shape (n, height, width) or (height, width)
        Returns:
            numpy.array: output data of shape (n, classes, height, width)
        """
        self.eval() # turns off dropout and batchnorm
        if im_path is not None:
            images = load_transforms({"img": im_path})
            images = images["img"]

        if images is None:
            raise ValueError("Either images or im_path must be specified")
        
        if isinstance(images, dict):
            images = images["img"]
        
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        
        if len(images.shape) == 2:
            images = images.unsqueeze(0)
        
        images = images.unsqueeze(1) # add batch dimension
        images = pre_transforms({"img": images})["img"]

        with torch.no_grad():
            outputs = self.inferer(images, self)
        
        outputs = post_transforms({"img": outputs})["img"]

        if outputs.shape[0] == 1:
            outputs = outputs.squeeze(0)

        if return_tensors == "pt":
            return outputs
        elif return_tensors == "np":
            return outputs.cpu().numpy()
        else:
            raise ValueError(f"return_tensors must be one of ['pt', 'np'], got {return_tensors}")

