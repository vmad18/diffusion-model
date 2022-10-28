import numpy as np
import torch

from utils.consts import *


class LocalPositionEmbedding(Module):

    r"""Local Position Embedding

    Local Position Information for Local Self Attention

    Args:
        ws: window size/local attention size

    """

    def __init__(self, ws: int):
        super().__init__()

        self.rel_pos = nn.Parameter(torch.randn(2*ws-1, 2*ws-1))
        self.pos = torch.tensor([(p1, p2) for p1 in range(ws) for p2 in range(ws)])
        self.pos = self.pos.unsqueeze(0) - self.pos.unsqueeze(1) + ws-1

    def forward(self, x: Tensor) -> Tensor:
        return x + self.rel_pos[self.pos[:, :, 0], self.pos[:, :, 1]]


class GlobalAttention(Module):

    r"""Multi-Head Self Attention

    Proposed in "Attention is All You Need"

    Args:
        dims: input data dimension
        dph: dimensions per head
        heads: number of heads

    """

    def __init__(self, dims: int, dph: int, heads: int):
        super().__init__()

        self.dims = dims
        self.h = heads
        self.pd = dph

        self.to_qkv = nn.Linear(dims, 3*dph*heads)
        self.proj = nn.Linear(dph*heads, dims)

    def forward(self, x: Tensor, mask=null) -> Tensor:
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).view(B, H*W, C)
        q, k, v = self.to_qkv(x).view(B, H*W, self.h, 3*self.pd).permute(0, 2, 1, 3).chunk(3, dim=-1)

        attn = (q @ k.transpose(-2, -1)) / np.sqrt(self.pd)

        if mask != null: attn = attn + mask * -1e9

        attn = F.softmax(attn, dim=-1)
        attn = attn @ v

        attn = attn.permute(0, 2, 1, 3).reshape(B, H*W, self.h*self.pd)
        return self.proj(attn).permute(0, 2, 1).view(B, C, H, W)


class LocalAttention(Module):

    r"""Local Multi-Head Self Attention

    Performs Multi-Head Self Attention On a Local mxm Region

    Args:
        dims: input data dimension
        dph: dimensions per head
        heads: number of heads
        ws: window size of the local region (mxm)

    """

    def __init__(self, dims: int, dph: int, heads: int, ws: int):
        super().__init__()

        self.dims = dims
        self.h = heads

        self.ns = dph
        self.ws = ws

        self.to_qkv = nn.Linear(dims, 3*dph*heads)
        self.proj = nn.Linear(dph*heads, dims)

        self.la = LocalPositionEmbedding(self.ws)

    def forward(self, x: Tensor, mask=null) -> Tensor:
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1)
        qkv = self.to_qkv(x).view(B, H, W, self.h, 3*self.ns).permute(0, 3, 1, 2, 4).view(B, self.h, H//self.ws, self.ws, W//self.ws, self.ws, 3*self.ns).permute(0, 1, 2, 4, 3, 5, 6)
        qkv = qkv.reshape(B, self.h, H//self.ws*W//self.ws, self.ws*self.ws, 3*self.ns).chunk(3, dim=-1)

        q, k, v = qkv
        attn = (q @ k.transpose(-2, -1))/np.sqrt(self.ns)

        if mask != null: attn = attn + mask * -1e9

        attn = F.softmax(self.la(attn), dim=-1)
        attn = attn @ v

        attn = attn.view(B, self.h, H//self.ws, W//self.ws, self.ws, self.ws, self.ns).permute(0, 2, 4, 3, 5, 1, 6).reshape(B, H, W, self.ns*self.h)
        return self.proj(attn).permute(0, 3, 1, 2)


if __name__ == "__main__":

    x: Tensor = torch.randn((3, 32, 56, 56))
    ga: GlobalAttention = GlobalAttention(32, 16, 8)
