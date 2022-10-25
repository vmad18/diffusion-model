from utils.consts import *
import copy

'''
Stochastic Weight Averaging Model Wrapper

Improves semi-supervised DL results through adding
gradient stability.

Algorithm: 

param_new = (decay) * param_new + (1-decay) * param_prev

https://arxiv.org/pdf/1703.01780.pdf

'''


class EMA(Module):

    def __init__(self,
                 model: Module,
                 alpha: float,
                 warm_up: int,
                 update_after: int,
                 gamma: int = 1.,
                 omega: int = -2/3,
                 device: str = "cuda"
                 ):

        super().__init__()

        self.model: Module = model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(false)

        self.start: bool = false

        self.alpha = alpha
        self.warm_up = warm_up
        self.update_after = update_after

        self.gamma = gamma
        self.omega = omega

        self.epoch: Tensor = torch.tensor([0]).to(device)

    @torch.no_grad()
    def update_alpha(self):
        point = torch.clamp(self.epoch - self.warm_up, 0.)
        self.alpha = torch.clamp(1. - torch.pow((1+point/self.gamma), self.omega), min=0., max=.9999)

    def apply(self):
        self.epoch += 1
        if self.epoch > self.warm_up and self.epoch%self.update_after == 0:

            if not self.start:
                for (ema_param, model_param) in zip(self.ema_model.parameters(), self.model.parameters()):
                    ema_param.data.copy_(model_param)
                self.start = true
                return
            self.average()

    @torch.no_grad()
    def average(self):
        self.update_alpha()
        for (model_param, ema_param) in zip(self.model.parameters(), self.ema_model.parameters()):

            shadow = ema_param.data - model_param.data
            shadow.mul_(1. - self.alpha)
            ema_param.sub_(shadow)

            ema_param.sub_((1.-self.alpha)*(ema_param-model_param))

    def get_model(self) -> Module:
        return self.ema_model


if __name__ == "__main__":
    net = nn.Linear(5, 10).cuda()
    ema = EMA(net, 0., 1, 2).cuda()

    for _ in range(10000):
        ema.apply()
        print(ema.alpha)
