from utils.consts import *
from models.Layers.Diffusion import NoiseScheduler, CosineScheduler
from UNet import UNet


def compute_loss(pred_noise: Tensor, noise: Tensor) -> Tensor:
    return F.mse_loss(pred_noise, noise, reduction="none").flatten(1).mean(1)


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
                 steps: int = 10000
                 ):

        super().__init__()

        self.steps = steps
        self.scheduler = schedule

        self.betas = self.scheduler.betas
        self.alphas = self.scheduler.alphas
        self.alpha_bars = self.scheduler.alpha_bars

        self.means = (torch.sqrt(torch.concat([torch.tensor([1.]).cuda(), self.alpha_bars[:-1]])) / (1. - self.alpha_bars))
        self.variance = ((1. - torch.concat([torch.tensor([1.]).cuda(), self.alpha_bars[:-1]])) / (1. - self.alpha_bars)) * self.betas

        # for predicting noise
        self.model = model

    """
    
    Sample from the forward posterior q(x_t | x0)
    
    """

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        return self.scheduler(x0, t, noise=noise)

    """
    
    Sample from the reverse posterior p(x_t-1 | x_t)
    
    returns x_t-1 , x_0, predicted noise 
    
    """
    def p_sample(self, x_t: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:

        noise_pred = self.predict_noise(x_t, t)
        mean_pred = expand(1./self.alphas[t].sqrt(), 3) * (x_t - expand(self.betas[t]/(1. - self.alpha_bars[t]).sqrt(), 3)*noise_pred)

        noise = noise_like(x_t, t)

        x_pt = mean_pred + expand(self.variance[t].sqrt(), 3) * noise

        return x_pt, self.scheduler(x_t, t, pred_noise=noise_pred), noise_pred

    def predict_noise(self, x_t: Tensor, t: Tensor):
        return self.model(x_t, t)

    r"""
    
    Args:
        x: Input image tensor
        t: Input time step tensor
        
    """
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        t = torch.randint(self.steps, (B,)).long().cuda()
        noise = torch.randn_like(x).cuda()

        degraded = self.q_sample(x, t, noise)
        x_pt, x0, noise_pred = self.p_sample(degraded, t)

        return compute_loss(noise_pred, noise)


if __name__ == "__main__":
    torch.cuda.set_device(torch.device("cuda:0"))
    diffusion = GaussianDiffusion(
        CosineScheduler(steps=10000).cuda(),
        UNet().cuda()
    )

    # print(diffusion(torch.randn(2, 3, 128, 128, dtype=torch.float64).cuda()))
