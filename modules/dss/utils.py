import torch
from torch.nn.functional import unfold


class LambdaPool2D(torch.nn.Module):
    def __init__(self, agg, kernel_size, stride=None, padding=0, dilation=1, return_indices=False,
                 ceil_mode=False):
        super(LambdaPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.agg = agg

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (b, c, h, w)
        Returns:
            torch.Tensor: output tensor of shape (b, c, h', w')
        """
        b, c, h, w = x.shape
        # unfold
        x = unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                   dilation=self.dilation)
        # aggregate on each kernel
        x = self.agg(x)

        # reshape
        h_out = (h + 2 * self.padding - self.dilation * (
                    self.kernel_size - 1) - 1) // self.stride + 1
        w_out = (w + 2 * self.padding - self.dilation * (
                    self.kernel_size - 1) - 1) // self.stride + 1
        x = x.view(b, c, h_out, w_out)
        return x


class ModePool2D(LambdaPool2D):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False,
                 ceil_mode=False):
        super(ModePool2D, self).__init__(agg=lambda x: torch.mode(x, dim=1)[0],
                                         kernel_size=kernel_size, stride=stride, padding=padding,
                                         dilation=dilation, return_indices=return_indices,
                                         ceil_mode=ceil_mode)
