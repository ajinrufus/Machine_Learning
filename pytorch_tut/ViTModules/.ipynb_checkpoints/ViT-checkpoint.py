import torch
from torch import nn
from ViTModules import transformer_enc, convFlat

class ViT(nn.Module):
    '''create vision transformer with ViT-Base hyperparamters'''

    def __init__(self, num_classes, 
                 img_size : int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_heads: int = 12,
                 atten_drop: float = 0,
                 mlp_size:int = 3072,
                 mlp_dropout: float =0.1,
                 num_transformer_layer: int = 12,
                 embedding_dropout:float = 0.1
                ):

        super().__init__()

        # number of patches
        N = int(img_size**2 / patch_size**2)

        # embedding size - all possible combinations of patches
        embed_size = patch_size**2 * in_channels

        # img to patches and flattening
        self.patch_embedding = convFlat.patchEmbed(in_channels=in_channels,
                                       embed_dim=embed_size,
                                       patch_size=patch_size)

        # class embedding (1 X D)
        self.class_embedding = nn.Parameter(torch.randn(1, 1, embed_size),
                                   requires_grad=True)

        # position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, N+1, embed_size),
                                          requires_grad=True)

        # transformer encoder
        self.transformer_encoder = nn.Sequential(*[transformer_enc.transformerEncoderBlock
                                                   (embed_size,num_heads,
                                                   atten_drop, mlp_size,
                                                   mlp_dropout) for _ in 
                                                   range(num_transformer_layer)])

        # create clasifier head
        self.classifier= nn.Sequential(nn.LayerNorm(normalized_shape=embed_size),
                                        nn.Linear(in_features=embed_size,
                                                out_features=num_classes))

        # create doreopout for embedding
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

    def forward(self, x):

        # batch size
        batch_size = x.shape[0]

        # repeat classes to all data
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        # patch embeedding
        x = self.patch_embedding(x)

        # concatenate class
        x = torch.cat((class_token, x), dim=1)

        # add position embedding
        x = self.pos_embedding + x

        # dropout embeedding (Appendix B.1)
        x = self.embedding_dropout(x)

        # transformer encoder
        x = self.transformer_encoder(x)

        # since class token output gives the class
        x = self.classifier(x[:, 0])

        return x
