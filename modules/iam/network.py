from monai.networks.nets import UNETR, UNet
from .config import SPATIAL_SIZE, N_CLASSES
import torch
import numpy as np
import os
from .transforms import load_transforms, pre_transforms
from monai.inferers import SlidingWindowInferer
from .utils import ModePool2D

_HIDDEN_SIZE = 768


class SegLMIAM(UNETR):
    def __init__(self, sw_batch_size=4, **kwargs):
        super().__init__(
            in_channels=2,
            out_channels=N_CLASSES,
            img_size=SPATIAL_SIZE,
            feature_size=32,
            num_heads=12,
            hidden_size=_HIDDEN_SIZE,
            mlp_dim=4 * _HIDDEN_SIZE,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.1,
            spatial_dims=2,
            conv_block=False,
            **kwargs
        )
        self.inferer = SlidingWindowInferer(roi_size=SPATIAL_SIZE, sw_batch_size=sw_batch_size,
                                            overlap=0.8)
        self.pool1 = ModePool2D(kernel_size=3, stride=1, padding=1)
        self.pool2 = ModePool2D(kernel_size=3, stride=1, padding=1)
        self.pool3 = ModePool2D(kernel_size=3, stride=1, padding=1)

    def __get_device(self):
        return next(self.parameters()).device

    def load(self, path):
        state = torch.load(os.path.join(os.getcwd(), path), map_location=torch.device('cpu'))
        self.load_state_dict(state)
        return self

    def predict(self, images=None, im_path=None, return_tensors="np", argmax=True):
        """
        Args:
            images (numpy.array): input data of shape (n, height, width) or (height, width)
        Returns:
            numpy.array: output data of shape (n, classes, height, width)
        """
        self.eval()  # turns off dropout and batchnorm

        if im_path is not None:
            # check if is dir
            if not os.path.isdir(im_path):
                images = load_transforms({"img": im_path})
                images = pre_transforms(images)
                images = images["img"]
                # print(images.shape, images.max(), images.min())
                # from matplotlib import pyplot as plt
                # plt.imshow(images[0])
                images = images.unsqueeze(0)
            else:
                image_paths = [os.path.join(im_path, f) for f in os.listdir(im_path)]
                images = [{"img": p} for p in image_paths]
                images = load_transforms(images)
                images = pre_transforms(images)
                images = [i["img"] for i in images]
                images = torch.stack(images)

        if images is None:
            raise ValueError("Either images or im_path must be specified")

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
            # add channel dimension
            images = images.unsqueeze(-3)
            if images.ndim == 3:
                images = images.unsqueeze(0)

        # precondition: images is a torch tensor of shape (n, 1, height, width)
        with torch.no_grad():
            images = images.to(self.__get_device())
            outputs = self.inferer(images, self)

        # post process
        if argmax:
            outputs = outputs.argmax(dim=1, keepdim=True).float()
            outputs = self.pool1(outputs)
            outputs = self.pool2(outputs)
            outputs = self.pool3(outputs)

        # remove channel dimension
        outputs = outputs.squeeze(-3)

        # Squeeze if only one image
        if outputs.shape[0] == 1:
            outputs = outputs.squeeze(0)

        if return_tensors == "pt":
            return outputs
        elif return_tensors == "np":
            return outputs.cpu().numpy()
        else:
            raise ValueError(f"return_tensors must be one of ['pt', 'np'], got {return_tensors}")


class LightSegLMIAM(UNet):
    def __init__(self, **kwargs):
        super().__init__(
            dimensions=2,
            in_channels=2,
            out_channels=N_CLASSES,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="instance",
            dropout=0.1,
            **kwargs
        )
        self.inferer = SlidingWindowInferer(roi_size=SPATIAL_SIZE, overlap=0.8)
        self.pool1 = ModePool2D(kernel_size=3, stride=1, padding=1)
        self.pool2 = ModePool2D(kernel_size=3, stride=1, padding=1)
        self.pool3 = ModePool2D(kernel_size=3, stride=1, padding=1)

    def __get_device(self):
        return next(self.parameters()).device

    def load(self, path):
        state = torch.load(os.path.join(os.getcwd(), path), map_location=torch.device('cpu'))
        self.load_state_dict(state)
        return self

    def predict(self, images=None, im_path=None, return_tensors="np", argmax=True):
        """
        Args:
            images (numpy.array): input data of shape (n, height, width) or (height, width)
        Returns:
            numpy.array: output data of shape (n, classes, height, width)
        """
        self.eval()  # turns off dropout and batchnorm

        if im_path is not None:
            # check if is dir
            if not os.path.isdir(im_path):
                images = load_transforms({"img": im_path})
                images = pre_transforms(images)
                images = images["img"]
                print(images.shape, images.max(), images.min())
                from matplotlib import pyplot as plt
                plt.imshow(images[0])
                images = images.unsqueeze(0)
            else:
                image_paths = [os.path.join(im_path, f) for f in os.listdir(im_path)]
                images = [{"img": p} for p in image_paths]
                images = load_transforms(images)
                images = pre_transforms(images)
                images = [i["img"] for i in images]
                images = torch.stack(images)

        if images is None:
            raise ValueError("Either images or im_path must be specified")

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
            # add channel dimension
            images = images.unsqueeze(-3)
            if images.ndim == 3:
                images = images.unsqueeze(0)

        # precondition: images is a torch tensor of shape (n, 1, height, width)
        with torch.no_grad():
            images = images.to(self.__get_device())
            outputs = self.inferer(images, self)

        # post process
        if argmax:
            outputs = outputs.argmax(dim=1, keepdim=True).float()
            outputs = self.pool1(outputs)
            outputs = self.pool2(outputs)
            outputs = self.pool3(outputs)

        # remove channel dimension
        outputs = outputs.squeeze(-3)

        # Squeeze if only one image
        if outputs.shape[0] == 1:
            outputs = outputs.squeeze(0)

        if return_tensors == "pt":
            return outputs
        elif return_tensors == "np":
            return outputs.cpu().numpy()
        else:
            raise ValueError(f"return_tensors must be one of ['pt', 'np'], got {return_tensors}")
