
from torch import nn

class MSABlock(nn.Module):
    '''
    Creates a multi-head self-attention block ("MSA block" for short).

    Args:
        embed_dim : size of embedded image
        num_heads  : number of items in parallel
        atten_drop : if dropout is required

    return:
        multihead attention output
    '''

    def __init__(self, embed_dim: int = 768,
                 num_heads : int = 12,
                 atten_drop : float =0):
        super().__init__()

        # 1. create layern norm
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)

        # 2. create MSA layer
        self.MSA = nn.MultiheadAttention(embed_dim=embed_dim,
                                         num_heads= 12, # parallel
                                         dropout=0, # no dropout is given
                                         batch_first=True)

    def forward(self, x):
        x = self.layer_norm(x)

        atten_out, _ = self.MSA(query = x,
                              key = x,
                              value = x,
                              need_weights = False) # require weights or only layer output

        return atten_out
