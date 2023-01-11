import gc

from utils.consts import *
from models.Layers.Diffusion import NoiseScheduler, CosineScheduler
from models.DiffusionModel.UNet import UNet
from tqdm import tqdm


def noise_like(x_t: Tensor, t: Tensor) -> Tensor:
    noises = torch.zeros_like(x_t)
    for i, j in enumerate(t):
        if j != 0:
            noises[i] = torch.randn(x_t.shape[1], x_t.shape[2], x_t.shape[3])
    return noises


class GaussianDiffusion(Module):

    r"""
    Performs the forward and reverse process.
    Forward process (q):
        Progressively degrades input by adding noise, up to a time step, until it is completely destroyed
    Reverse process (p):
        Predicts the previous time step by predicting and subtracting the noise of the current step.
        Predicts noise via parametrized model.
    """

    def __init__(self,
                 schedule: NoiseScheduler,
                 model,
                 channels: int = 3,
                 img_d: int = 64,
                 device: str = "cuda",
                 steps: int = 1000
                 ):

        super().__init__()

        self.steps = steps
        self.scheduler = schedule

        self.betas = self.scheduler.betas
        
        self.alphas = self.scheduler.alphas
        self.alpha_bars = self.scheduler.alpha_bars
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value = 1.)
        
        self.means1 = self.betas * torch.sqrt(self.alpha_bars_prev) / (1. - self.alpha_bars)
        
        self.means2 = (1. - self.alpha_bars_prev) * self.alphas.sqrt() / (1. - self.alpha_bars)
        
        #self.means = (torch.sqrt(torch.concat([torch.tensor([1.], device=device), self.alpha_bars[:-1]])) / (1. - self.alpha_bars))
        
        
        self.variance = self.betas * (1. - self.alpha_bars_prev)/(1. - self.alpha_bars)
        
        #((1. - torch.concat([torch.tensor([1.], device=device), self.alpha_bars[:-1]])) / (1. - self.alpha_bars)) * self.betas

        # for predicting noise
        self.model = model

        self.channels = channels
        self.image_size = img_d

        self.device = device

    """
    
    Forward posterior q(x_t | x0)
    
    """

    def q_post(self, x_0: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        post_mean = expand(self.means1[t], 3) * x_0 + expand(self.means2[t], 3) * x_t
        
        return post_mean 
        
        
        
    """
    
    Sample from the forward posterior q(x_t | x0)
    
    """

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        return self.scheduler(x0, t, noise=noise)
    
    """
    
    Sample from the reverse posterior p(x_t-1 | x_t)
    returns x_t-1 , x_0, predicted noise 
    
    """

    @torch.no_grad()
    def p_sample(self, x_t: Tensor, t: Tensor, ts: int):
        
        noise_pred = self.predict_noise(x_t, t)
        
        x_0 = torch.clip(self.scheduler(x_t, t, pred_noise=noise_pred), -1, 1)
        
        mean_pred = self.q_post(x_0, x_t, t)
        
        #mean_pred = expand((1./self.alphas[t]).sqrt(), 3) * (x_t - expand(self.betas[t]/(1. - self.alpha_bars[t]).sqrt(), 3)*noise_pred)

        noise = torch.randn_like(x_t) if ts > 0 else 0.

        x_pt = mean_pred + expand(self.variance[t].sqrt(), 3) * noise
        return x_pt, x_0, noise_pred

    """
    
    Reverse markov chain process
    p(x_(t-1) | x_t)
    
    """

    @torch.no_grad()
    def markov(self, shape):

        b = shape[0]

        img = torch.randn(shape, device=self.device)

        for t in tqdm(reversed(range(0, self.steps)), desc = "sample loop", total = self.steps):
            ts = torch.full((b,), t, dtype=torch.long, device=self.device)
            img, x_0, noise = self.p_sample(img, ts, t)

        return img

    def predict_noise(self, x_t: Tensor, t: Tensor):
        return self.model(x_t.to("cuda", dtype=torch.float), t)

    """
    
    Sampling function for DDPM
    
    """

    @torch.no_grad()
    def sample(self, bs: int = 4):
        return self.markov((bs, self.channels, self.image_size, self.image_size))

    """
    Args:
        pred_noise: Predicted noise
        noise: Ground truth noise
    """

    def compute_loss(self, pred_noise: Tensor, noise: Tensor) -> Tensor:
        return F.mse_loss(pred_noise, noise)

    """
    
    Args:
        x: Input image tensor
        t: Input time step tensor
        
    """

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        t = torch.randint(self.steps, (B,)).long().cuda()
        noise = noise_like(x, t).cuda()

        degraded = self.q_sample(x, t, noise)
        noise_pred = self.predict_noise(degraded, t)

        return self.compute_loss(noise_pred, noise)


if __name__ == "__main__":
    torch.cuda.set_device(torch.device("cuda"))
    diffusion = GaussianDiffusion(
        CosineScheduler(steps=10000).cuda(),
        UNet().cuda()
    )
