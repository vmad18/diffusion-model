from utils.consts import *
from utils.Layers import GlobalAttention, LocalAttention
from models.Layers.Diffusion import TimeEncoding, TimeProjection, ResNetBlock, EmbeddingPass


class UNet(Module):

    r"""UNet With Attention

    Args:
        dims: projected feature dimension
        channel_scale: feature dimension scale per depth
        attn_res: resolution where attention layers are used
        blocks: number of resnet blocks before up/down sampling
        scale: the scale to the time encoding dimension


    """

    def __init__(self,
                 dims: int = 128,
                 channel_scale: list = (1, 2, 3, 4),
                 attn_res: int = 256,
                 blocks: int = 8,
                 scale: int = 4
                 ):

        super().__init__()

        self.t_enc = TimeEncoding(dims)
        self.t_embedding = TimeProjection(dims, scale*dims, embed=true)

        dim_layer = [(dims*channel_scale[i-1], dims*channel_scale[i]) for i in range(1, len(channel_scale))]
        dim_layer.insert(0, (dims, dims*channel_scale[0]))
        dim_layer = np.asarray(dim_layer)

        self.inp_layers = nn.Sequential(nn.Conv2d(3, dims, 3, padding=1), nn.SiLU(inplace=true))

        self.down_blocks = nn.ModuleList([])
        for di, do in dim_layer:
            layers = EmbeddingPass()
            for i in range(blocks):
                if i != blocks-1:
                    layers.append(ResNetBlock(dim_in=di, dim_out=di, dim_t=scale*dims))
                else:
                    layers.append(ResNetBlock(dim_in=di, dim_out=do, dim_t=scale*dims, down=true))

            if do == attn_res:
                layers.append(GlobalAttention(do, 64, 4))

            self.down_blocks.append(layers)

        mid_dim = dim_layer[-1][1]
        self.middle_block = EmbeddingPass(
            ResNetBlock(dim_in=mid_dim, dim_out=mid_dim, dim_t=scale*dims),
            GlobalAttention(mid_dim, 64, 4),
            ResNetBlock(dim_in=mid_dim, dim_out=mid_dim, dim_t=scale*dims)
        )

        self.up_blocks = nn.ModuleList([])
        for di, do in dim_layer[::-1]:
            layers = EmbeddingPass()
            for i in range(blocks):
                if i == 0:
                    layers.append(ResNetBlock(dim_in=do+do, dim_out=do, dim_t=scale*dims))
                elif i != blocks-1:
                    layers.append(ResNetBlock(dim_in=do, dim_out=do, dim_t=scale*dims))
                else:
                    layers.append(ResNetBlock(dim_in=do, dim_out=di, dim_t=scale*dims, up=true))

            if di == attn_res:
                layers.append(GlobalAttention(di, 64, 4))

            self.up_blocks.append(layers)

        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(dims, 3, kernel_size=3, padding=1))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:

        t_embed = self.t_embedding(self.t_enc(t))

        jumps = []

        x = self.inp_layers(x.float())

        for module in self.down_blocks:
            x = module(x, t_embed)
            jumps.append(x)

        x = self.middle_block(x, t_embed)

        for module in self.up_blocks:
            x = torch.concat([jumps.pop(), x], dim=1)
            x = module(x, t_embed)

        return self.out_layers(x)


if __name__ == "__main__":
    unet = UNet()
    tnsr = torch.randn((1, 3, 128, 128))
    print(unet(tnsr, torch.tensor([1])))

''''
U-Net
input layer d_in (3) -> dims
ResBlocks n dims -> 2*dims 
ResBlocks n 2*dims -> 3*dims 
ResBlocks n 3*dims -> 4*dims 
ResBlocks n 4*dims -> 3*dims 
ResBlocks n 3*dims -> 2*dims
ResBlocks n 2*dims -> dims 
output layer dims -> d_in (3)
'''
