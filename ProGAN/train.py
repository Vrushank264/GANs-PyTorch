import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from math import log2
from utils import *
import utils
from Models import Generator, Discriminator
import config
import torchvision.utils as vutils
from scipy.stats import truncnorm

torch.backends.cudnn.benchmarks = True
G_l = []
D_l = []


def data_loader(img_size):
    transform = transforms.Compose( 
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.25),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    batch_size = config.batch_sizes[(int(log2(img_size / 4)))]
    dataset = datasets.ImageFolder(
        root=config.root, transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return loader, dataset


def train(disc,gen,loader,dataset,step,alpha,opt_d,opt_g,tensorboard_step,writer,scalar_disc,scalar_gen):

    loop = tqdm(loader, leave=True, position=0)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.device)
        cur_batch_size = real.shape[0]
        noise = torch.randn((cur_batch_size, config.latent_dim, 1, 1)).to(config.device)

        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            disc_real = disc(real, alpha, step)
            disc_fake = disc(fake.detach(), alpha, step)
            gp = gradient_penalty(disc, real, fake, alpha, step, device=config.device)
            d_loss = (
                -(torch.mean(disc_real) - torch.mean(disc_fake))+ config.lambda_gp * gp + (0.001 * torch.mean(disc_real ** 2))
            )
          
        opt_d.zero_grad()
        scalar_disc.scale(d_loss).backward()
        scalar_disc.step(opt_d)
        scalar_disc.update()

        with torch.cuda.amp.autocast():
            gen_fake = disc(fake, alpha, step)
            g_loss = -torch.mean(gen_fake)
            
        opt_g.zero_grad()
        scalar_gen.scale(g_loss).backward()
        scalar_gen.step(opt_g)
        scalar_gen.update()

        alpha += cur_batch_size / (config.progressive_epochs[step] * 0.5) * len(dataset)
        alpha = min(alpha, 1)
        G_l.append(g_loss.item())
        D_l.append(d_loss.item())

        if batch_idx % 100 == 0:

            with torch.no_grad():
                fake_imgs = gen(config.fixed_noise, alpha, step) * 0.5 + 0.5
                plot_tensorboard(
                    writer,
                    d_loss.item(),
                    g_loss.item(),
                    real.detach(),
                    fake_imgs.detach(),
                    tensorboard_step,
                )
                tensorboard_step += 1

            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator loss")
            plt.plot(G_l, label="Generator")
            plt.plot(D_l, label="Discriminator")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
            loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item(), gp=gp.item())

    return tensorboard_step, alpha


def main():

    gen = Generator(config.latent_dim, config.in_c, img_c=3).to(config.device)
    #gen.load_state_dict(torch.load('E:/Computer Vision/ProGAN/Saved Models/Generator.pth'))
    disc = Discriminator(config.in_c, img_c=3).to(config.device)
    #disc.load_state_dict(torch.load('E:/Computer Vision/ProGAN/Saved Models/Discriminator.pth'))
    opt_g = optim.Adam(gen.parameters(), lr=config.lr, betas=(0.0, 0.99))
    opt_d = optim.Adam(disc.parameters(), lr=config.lr, betas=(0.0, 0.99))

    writer = SummaryWriter("logs/proGAN")

    gen.train()
    disc.train()
    scalar_gen = torch.cuda.amp.GradScaler()
    scalar_disc = torch.cuda.amp.GradScaler()
    tensorboard_step = 0
    step = int(log2(128 / 4))

    for epochs in config.progressive_epochs[step:]:

        alpha = 1e-5
        loader, dataset = data_loader(4 * 2 ** step)
        print(f"Current Image Size: {4*2**step}")

        for epoch in range(epochs):
            print(f"Epoch: [{epoch+1}/{epochs}]")
            tensorboard_step, alpha = train(disc,gen,loader,dataset,step,alpha,opt_d, opt_g,tensorboard_step,writer,scalar_disc,scalar_gen)
            if epoch % 1 == 0:
            
                torch.save(
                    gen.state_dict(),
                    open(config.save_path + "Generator.pth", "wb"),
                )
                torch.save(
                    disc.state_dict(),
                    open(config.save_path + "Discriminator.pth", "wb")
                )

        step += 1



if __name__ == "__main__":

    main()
    '''gen = Generator(config.latent_dim, config.in_c, img_c=3).to(config.device)
    #gen.load_state_dict(torch.load('E:/Computer Vision/ProGAN/Saved Models/Generator_128_final.pth'))
    step = int(log2(128 / 4))
    alpha = 1.0
    truncation = 0.7
    for i in range(64):
        with torch.no_grad():
            noise = torch.tensor(
                truncnorm.rvs(
                    -truncation, truncation, size=(1, config.latent_dim, 1, 1)
                ),
                device=config.device,
                dtype=torch.float32,
            )
            img = gen(noise, alpha, step)
            vutils.save_image(img * 0.5 + 0.5, f"E:/Computer Vision/ProGAN/generated_images/img{i}.png")
    gen.train()
    print('Success!')'''
    
