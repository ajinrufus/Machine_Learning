from torch import nn
from ViTModules import MSA, MLP

class transformerEncoderBlock(nn.Module):

    def __init__(self, embed_dim: int = 768,
                 num_heads : int = 12,
                 atten_drop : float =0,
                 mlp_size : int = 3072, # given in table 1
                 mlp_dropout : float =0.1):

        '''creates a transformer encoder block
        
        Args:
            embed_dim : size of embedded image
            num_heads  : number of items in parallel
            atten_drop : if dropout is required
            mlp_size  : size of hidden units
            dropout : Amount of droput required

        returns:
            ViT transformer encoded
        '''

        super().__init__()

        # msa block
        self.msa_block = MSA.MSABlock(embed_dim = embed_dim,
                                  num_heads = num_heads,
                                  atten_drop = atten_drop)

        # mlp block
        self.mlp_block = MLP.MLPBlock(embed_dim=embed_dim, mlp_size=mlp_size,
                             dropout=mlp_dropout)


    def forward(self, x):

        # msa with resiudal connection added (input added)
        x = self.msa_block(x) + x

        # mlp with resiudal connection added (input added)
        x = self.mlp_block(x) + x

        return x
