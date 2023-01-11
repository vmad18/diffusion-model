import sys
#sys.path.append("/home/vd71/fs/diffusion-main/utils")

from utils.consts import *
#from utils.dataset_test import CustomImageDataset

from models.DiffusionModel.Training import *
from models.DiffusionModel.UNet import UNet
from models.DiffusionModel.DiffusionModel import GaussianDiffusion, CosineScheduler
from torchvision.datasets import CIFAR10, StanfordCars, Flowers102
from torchvision import transforms


def main(model_num: int = -1, img_d: int = 32, dims: int = 64):
    transform = transforms.Compose(
        [transforms.Resize((img_d, img_d)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = Flowers102(root='./data', transform=transform, download=true)
    # dataset = StanfordCars(root='./data2', transform=transform, download=true)
    # loader = DataLoader(dataset, batch_size=16, shuffle=true, drop_last=true, num_workers=4)

    #dataset = CustomImageDataset("ubiquitous-pancake/utils/to_train.csv", "ubiquitous-pancake/utils/save")
    loader = DataLoader(dataset, batch_size=32, shuffle=true, num_workers=4)
    
    # dataset = CIFAR10(root='./data', train=true, transform=transform, download=true)
    # loader = DataLoader(dataset, batch_size=16, shuffle=true, num_workers=4)

    gd = GaussianDiffusion(
        CosineScheduler(steps=int(1e3)),
        UNet(dim=dims).cuda(),
        channels=3,
        img_d=img_d,
        steps=int(1e3)
    ).cuda()

    trainer = Train(1000, gd, last_epoch=model_num, load_epoch=model_num, i_dim=img_d, parallel=false)
    trainer.train(loader)


def main2(model_num: int = -1, img_d: int = 32, dims: int = 64):

    num: int = int(input("Picture Number: "))
        
    gd = GaussianDiffusion(
        CosineScheduler(steps=int(1e3)),
        UNet(dim=dims).cuda(),
        channels=3,
        img_d=img_d,
        steps=int(1e3)
    ).cuda()

    trainer = Train(1000, gd, load_epoch=model_num, i_dim=img_d)
    trainer.plot_sampled(num)


if __name__ == "__main__":    
    test: str = input("Test mode? ").lower()

    args = [-1, 64, 128]

    if test in ('y', "yes", "test"):
        print("Testing...")
        main2(args[0], args[1], args[2])
    elif test in ('n', "no", "train"):
        main(args[0], args[1], args[2])
    else:
        print("can't do this")


# if __name__ == "__main__":
#     model = UNet().cuda()
#     gd = GaussianDiffusion(
#         CosineScheduler(steps=int(1e3)),
#         model=model,
#         img_d=128,
#         steps=int(1e3)
#     ).cuda()
#
#     print(gd(torch.randn((3, 3, 128, 128)).cuda()))
