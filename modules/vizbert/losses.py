import torch
import torch.nn.functional as F


# Taken from https://innovationincubator.com/siamese-neural-network-with-pytorch-code-example/
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label=None, equal=True):
        # Find the pairwise distance or eucledian distance of two output feature vectors
        euclidean_distance = F.pairwise_distance(output1, output2)
        if equal:
            return torch.mean(torch.pow(euclidean_distance, 2))
        # perform contrastive loss calculation with the distance
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
