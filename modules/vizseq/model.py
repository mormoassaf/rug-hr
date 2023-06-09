# VizSeq 
################################################################################
# Network architecture created by Mo Assaf (moassaf42@gmail.com)               #
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import VizSeqEncoder


class VizSeqMasked(torch.nn.Module):

    def __init__(self,
                 max_seq_length=128,
                 spatial_size=(128, 2048),
                 vocab_size=30522,
                 in_channels=1,
                 num_heads=12,
                 num_layers=12,
                 hidden_size=768,
                 mlp_hidden_size=3072,
                 dropout=0.1
                 ):
        super(VizSeqMasked, self).__init__()
        self.max_seq_length = max_seq_length
        self.spatial_size = spatial_size
        self.vocab_size = vocab_size
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Network architecture
        ## (batch, in_channels, spatial_size[0], spatial_size[1]) -> (batch, sequence_length, embedding_size)
        self.feature_extractor = VizSeqEncoder(
            embedding_size=hidden_size,
            max_sequence_length=max_seq_length,
            in_channels=in_channels,
            spatial_size=spatial_size)
        ## (batch, max_seq_length, embedding_size) -> (batch, sequence_length, vocab_size)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=mlp_hidden_size,
                dropout=dropout,
                activation='gelu'
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size)
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, vocab_size),
        )

    """
        Feeds the image through the network and returns the logits for each token in the sequence.

        image: (batch_size, in_channels, *spatial_size)
        mask: (batch_size, *spatial_size)

        return: (batch_size, max_seq_length, vocab_size)
    """

    def forward(self, image, mask=None):
        if not mask == None:
            masked_image = image * mask.unsqueeze(1)
        else:
            masked_image = image
        encoding = self.feature_extractor(masked_image)
        encoding = encoding.transpose(0, 1)
        decoding = self.decoder(
            tgt=encoding,
            memory=encoding,
        )
        decoding = decoding.transpose(0, 1)
        decoding = self.out(decoding)
        return decoding
