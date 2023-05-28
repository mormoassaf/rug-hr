
import torch
from torchvision import models
from .unet_basic_encoder import UNetBasicEncoder

class VizBERTEncoder(torch.nn.Module):

    def __init__(self, 
                 embedding_size=768,
                 max_sequence_length=256,
                 in_channels=1,
                 spatial_size=(128, 2048),
                 device=torch.device("cpu"),
                 ):
        super().__init__()
        self.device = device
        self.spatial_size = spatial_size
        self.encoder = UNetBasicEncoder(in_channels=in_channels)
        # output size is (batch, last_channel=1024, spatial_size[0]//2**4, spatial_size[1]//2**4)
        #  output of encoding2sequence goal (batch, sequence_length, embedding_size)
        out_h = spatial_size[0]//(2**4)
        out_w = spatial_size[1]//(2**4)
        self.encoding2embedding = torch.nn.Sequential(
            torch.nn.Conv2d(1024, max_sequence_length, kernel_size=(1, 1)), # outputs (batch, embedding_size, out_h, out_w)
            torch.nn.SiLU(), # outputs (batch, max_sequence_length, out_h, out_w)
            torch.nn.Flatten(start_dim=2), # outputs (batch, max_sequence_length, out_h*out_w)
            torch.nn.Linear(out_h * out_w, embedding_size), # outputs (batch, sequence_length, embedding_size)
        )

        # resnet = models.resnet50(pretrained=True)
        # layers = list(resnet.children())  # remove the last layer
        # layers[0] = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # layers = layers[:-3]
        # self.encoder = torch.nn.Sequential(*layers)
        # self.encoding2embedding = torch.nn.Sequential(
        #     torch.nn.Conv2d(1024, max_sequence_length, kernel_size=(1, 1)), # outputs (batch, max_sequence_length, 8, 128)
        #     torch.nn.SiLU(), # outputs (batch, embedding_size, 8, 128)
        #     torch.nn.Flatten(start_dim=2), # outputs (batch, max_sequence_length, 8*128)
        #     torch.nn.Linear(8*128, embedding_size), # outputs (batch, max_sequence_length, vocab_size)
        # )


    """
    image: (batch_size, in_channels, *spatial_size)
    timestep: (batch_size, 1)

    """
    def forward(self, image):

        encoding = self.encoder(image)
        encoding = self.encoding2embedding(encoding)

        return encoding
