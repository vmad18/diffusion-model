from utils.consts import *
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


class NoiseScheduler(Module):

    r"""

    Progressively adds more noise until input image gets
    destroyed (becomes an isotropic gaussian)

    Args:
        steps: number of steps before more or less a full isotropic gaussian is achieved


    """

    def __init__(self, steps: int, device: str = "cuda", *args, **kwargs):

        super().__init__()

        self.steps = steps

        self.device = device

        self.betas = null
        self.alphas = null
        self.alpha_bars = null

    """
    x_t = sqrt(a_bar) * x_0 + sqrt(1 - a_bar) * N(0, 1)
    
    :param x - input to add/remove noise from
    :param t - time step
    :param forward - default true if forward process
    :param pred_noise - predicted noise to remove in reverse process
    
    :returns noised image at a time step, t, from x_0
    """

    def forward(self, x: Tensor, t: Tensor, noise=null, pred_noise=null) -> Tensor:
        if noise != null:
            x0 = x # start input
            return expand(self.alpha_bars[t].sqrt(), 3)*x0 + expand(torch.sqrt(1. - self.alpha_bars[t]), 3) * noise
        else:
            x_pt = x # previous time step
            return expand(1./torch.sqrt(self.alpha_bars[t]), 3) * (x_pt - expand(torch.sqrt(1. - self.alpha_bars[t]), 3) * pred_noise)


class CosineScheduler(NoiseScheduler):

    r"""Cosine Scheduler: Improves FID score over Linear Scheduling

    Args:
        steps: number of time steps
        s: offset value

    """

    def __init__(self, s: float = .008, **kwargs):
        super().__init__(**kwargs)

        self.s = s

        self.betas = self.compute().to(self.device)
        self.alphas = 1. - self.betas

        self.alpha_bars = torch.cumprod(self.alphas, dim=-1)

    def compute(self) -> Tensor:

        ts = torch.linspace(0, self.steps, self.steps+1, dtype=torch.float64)
        alpha_bars = ((ts/self.steps + self.s)/(1. + self.s) * torch.pi/2.).cos() ** 2

        alpha_bars = alpha_bars/alpha_bars[0]

        betas = 1. - (alpha_bars[1:]/alpha_bars[:-1])
        return torch.clip(betas, min=0., max=.999)


class UpSample2D(Module):

    r"""Wrapper class for interpolation (equivalent effect to Upsample2D module)

    Args:
        scale_factor: factor to scale dimensions by

    """

    def __init__(self, scale_factor: int = 2):
        super().__init__()

        self.sf = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        return F.interpolate(x, scale_factor=self.sf, mode="nearest")


class TimeEncoding(Module):

    r"""Positional encoding algorithm as proposed in "Attention is All You Need"

    Args:
        dims: dimension to encode time information

    """

    def __init__(self, dims: int):
        super().__init__()

        self.dims = dims
        self.divs = torch.exp(torch.arange(0, dims, 2) * -np.log(1e4)/dims)

    def forward(self, t: Tensor) -> Tensor:
        device = t.device
        self.divs = self.divs.to(device)
        encodes = torch.zeros((t.shape[0], self.dims), device=device)
        encodes[:, 0::2] = (t.unsqueeze(1) * self.divs).sin()
        encodes[:, 1::2] = (t.unsqueeze(1) * self.divs).cos()
        return encodes


class TimeProjection(Module):

    r"""Module for either projecting or embedding time information

    Args:
        dims: input dimension
        dim_out: projected dimension
        nl: non-linearity, default SiLU
        embed: embed input with linear transformation

    """

    def __init__(self,
                 dims: int,
                 dim_out: int,
                 nl=F.silu,
                 embed: bool = false):

        super().__init__()

        self.nl = nl
        self.embed = nn.Linear(dims, dim_out) if embed else nn.Identity()
        self.proj = nn.Linear(dim_out, dim_out) if embed else nn.Linear(dims, dim_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.nl(self.embed(x)))


class ResNetBlock(Module):

    r"""BigGan ResNet Block

    Args:
        dim_in: the in dimensions
        dim_out: the out dimensions
        dim_t: the time embedding dimensions
        dp: dropout rate
        nl: non-linearity function (GELU for better convergence)
        up: up-sample input
        down: down-sample input

    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_t: int,
                 dp: float = .1,
                 nl=nn.GELU(),
                 up: bool = false,
                 down: bool = false):

        super().__init__()

        self.di = dim_in
        self.do = dim_out

        self.t_proj = TimeProjection(dim_t, dim_out)

        dropout = nn.Dropout(dp)

        if up:
            scale = UpSample2D()
        elif down:
            scale = nn.AvgPool2d(2, 2)
        else:
            scale = nn.Identity()

        self.in_transform = nn.Sequential(
            nn.GroupNorm(32, num_channels=dim_in),
            nl,
            dropout,
            scale,
            nn.Conv2d(dim_in, dim_out, 3, padding=1)
        )

        self.out_transform = nn.Sequential(
            nn.GroupNorm(32, num_channels=dim_out),
            nl,
            dropout,
            nn.Conv2d(dim_out, dim_out, 3, padding=1)
        )

        if up or down:
            self.skip_con = nn.Sequential(
                scale,
                nn.Conv2d(dim_in, dim_out, 3, padding=1)
            )
        else:
            self.skip_con = nn.Sequential(
                scale,
                nn.Conv2d(dim_in, dim_out, 1)
            )

    r"""
    
    Args:
        x - input image tensor
        t - input time embedding tensor
    
    """

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x2 = x.clone()
        x = self.in_transform(x) + self.t_proj(t)[..., None, None]

        return self.skip_con(x2) + self.out_transform(x)


class EmbeddingPass(nn.Sequential):

    def forward(self, x: Tensor, t: Tensor) -> Tensor:

        for layer in self:
            if type(layer) == ResNetBlock:
                x = layer(x, t)
            else:
                x = layer(x)

        return x


if __name__ == "__main__":
    # te: TimeEncoding = TimeEncoding(128)
    # print(te(torch.randint(10, size=(4,))).shape)

    tnsr = torch.randn(1, 128, 64, 64)

    t_enc = TimeEncoding(128)
    t_emb = TimeProjection(128, 4*128, embed=true)

    t_emb = t_emb(t_enc(torch.tensor([1])))

    rnb: ResNetBlock = ResNetBlock(128, 64, 128*4, up=true)

    print(rnb(tnsr, t_emb).shape)

    # cs: CosineScheduler = CosineScheduler(steps=1000)
    #
    # img: Image = Image.open("face.png").convert('RGB').resize((256, 256))
    #
    # tnsr = torch.tensor(np.asarray(img.getdata()).reshape(img.size[1], img.size[0], 3)).permute(2, 0, 1).unsqueeze(0).double()/255. * 2. - 1.
    # tnsr = (cs(tnsr, 0).permute(0, 2, 3, 1)[0] + 1.) / 2. * 255.
    #
    # img: Image = Image.fromarray(np.uint8(tnsr)).convert('RGB')
    # img.show()




    # i = cs(torch.zeros(3, 3, 64, 64), 99)
    # i = i.permute(0, 2, 3, 1)[0]

    # trans = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    #
    # dataset = CIFAR10(root="./data", transform=trans, download=true)
    # loader = DataLoader(dataset, batch_size=4, shuffle=true)
    #
    # print(torch.tensor(loader.dataset[0][0]))
