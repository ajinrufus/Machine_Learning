
from torch import nn

class MLPBlock(nn.Module):
    '''
    Creates a multilayer perceptron block ("MLP block" for short).

    Args:
        embed_dim : size of embedded image
        mlp_size  : size of hidden units
        dropout : Amount of droput required

    return:
        multilayer perceptron passed output
    '''

    def __init__(self, embed_dim: int = 768,
                 mlp_size : int = 3072, # given in table 1
                 dropout : float =0.1):
        super().__init__()

        # 1. create layern norm
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)

        # 2. create MSA layer
        self.MLP = nn.Sequential(nn.Linear(in_features=embed_dim,
                                           out_features=mlp_size), # mentioned in table 1
                                 nn.GELU(),
                                 nn.Dropout(p=dropout),
                                nn.Linear(in_features=mlp_size,
                                           out_features=embed_dim),
                                 nn.Dropout(p=dropout)
                                )
            
    def forward(self, x):
        x = self.layer_norm(x)

        x = self.MLP(x)

        return x
