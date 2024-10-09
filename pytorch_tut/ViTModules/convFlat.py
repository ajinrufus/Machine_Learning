
from torch import nn

class patchEmbed(nn.Module):
    '''Turn 2D image into a 1D sequence of learnable vector

    Args:
        in_channels: Number of color channels of input image
        embed_dim: embedding size required
        patch_size  : size of each patch
    '''

    def __init__(self, in_channels: int = 3,
                 embed_dim: int = 768, patch_size: int= 16):
        super().__init__()

        self.patch_size = patch_size
        
        # 1. img to patches
        self.conv2d = nn.Conv2d(in_channels = in_channels,
                           out_channels= embed_dim,
                           kernel_size=self.patch_size,
                           stride = self.patch_size,
                           padding= 0)

        # 2. flatten to single dimension like text dimension req for transformer
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):

        # input size must be divisible by patch size
        img_res = x.shape[-1]
        assert img_res % self.patch_size == 0

        #patching
        x_patched = self.conv2d(x)

        # flattening
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0,2,1) # to required shape
