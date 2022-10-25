from utils.consts import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.EMA.EMA import EMA
from models.DiffusionModel.DiffusionModel import GaussianDiffusion, UNet, CosineScheduler
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image


ddpm_path: str = "./ddpm_"
ema_model: str = "./ema_model_"


class Train:

    def __init__(self,
                 epochs: int,
                 diffuser: Module,
                 device: str = "cuda",
                 load_epoch: int = -1
                 ):

        self.epochs = epochs

        self.ddpm = diffuser
        self.ema_model = EMA(diffuser, 0.0, 100, 10)

        if load_epoch != -1:
            self.ddpm.load_state_dict(torch.load(ddpm_path+str(load_epoch)+".pth"))
            self.ema_model.load_state_dict(torch.load(ema_model+str(load_epoch)+".pth"))

        self.opt = Adam(diffuser.parameters(), lr=1e-4)

        self.device = device

    def plot_sampled(self):
        img = torch.randn((1, 3, 32, 32), device=self.device)
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        num_images = 10
        stepsize = int(1e3/num_images)

        for i in range(0, int(1e2))[::-1]:
            t = torch.tensor([i], dtype=torch.long, device=self.device)
            x_p, x_0, noise = self.ddpm.p_sample(img, t)
            x_p = x_0.detach().to("cpu")
            if i % stepsize == 0:
                plt.subplot(1, num_images, int(i/stepsize+1))

                x_p = 255. * (x_p + 1)/2
                x_p = Image.fromarray(x_p[0].permute(1, 2, 0).numpy().astype(np.uint8))

                plt.imshow(x_p)

        plt.show()

    def train(self, dl: DataLoader):

        for _ in range(self.epochs):
            self.ddpm.train()
            total_loss: float = 0.0
            data = iter(dl)
            pbar = tqdm(data, desc="Mini-Batch", position=0, leave=false)
            for i, batch in enumerate(pbar):

                x = batch[0].to(self.device)
                loss = self.ddpm(x)

                total_loss += loss.sum(0).item()

                self.opt.zero_grad()
                loss.sum(0).backward() # compute loss gradient
                self.opt.step() # back propagate and gradient descent

                self.ema_model.apply()
                pbar.set_postfix({"Loss" : loss.sum(0)/100})

                # if (i+1) % 1e3 == 0:
                #     self.plot_sampled()

            print(f"TOTAL LOSS: {total_loss}")
            if _ % 5 == 0:
                print("saving checkpoint")
                torch.save(self.ddpm.state_dict(), f"./{ddpm_path}{_}.pth")
                torch.save(self.ema_model.state_dict(), f"./{ema_model}{_}.pth")




if __name__ == "__main__":
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = CIFAR10(root='./data', train=true, transform=transform)

    loader = DataLoader(dataset, batch_size=16, shuffle=true, num_workers=4)
    gd = GaussianDiffusion(
        CosineScheduler(steps=int(1e2)).cuda(),
        UNet(dims=32).cuda(),
        steps=int(1e2)
    )

    trainer = Train(1000, gd, load_epoch=-1)
    #trainer.plot_sampled()
    trainer.train(loader)
