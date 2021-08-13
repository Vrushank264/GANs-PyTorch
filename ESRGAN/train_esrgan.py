import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter
import config
from models import Generator, Discriminator, init_weights
from dataset import TrainDataset

class ContentLoss(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.vgg19 = models.vgg19(pretrained = True).features[:35].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()
        for params in self.vgg19.parameters():
            params.requires_grad =False
            
    def forward(self, ip_img: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        
        ip_features = self.vgg19(ip_img)
        target_features = self.vgg19(target_image)
        vgg_loss = self.loss(ip_features, target_features)
        return vgg_loss


def gradient_penalty(disc, real, fake):
    
    batch_size, c, h, w = real.shape
    alpha = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(config.DEVICE)
    interpolated_imgs = (real * alpha) + (fake.detach() * (1 - alpha))
    interpolated_imgs.requires_grad_(True)
    
    disc_scores = disc(interpolated_imgs)
    
    gradient = torch.autograd.grad(inputs = interpolated_imgs, outputs = disc_scores, grad_outputs = torch.ones_like(disc_scores), 
                                   create_graph = True, retain_graph = True)[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    norm = gradient.norm(2, dim = 1)
    penalty = torch.mean((norm - 1) ** 2)
    
    return penalty

    
def psnrTrain(loader, gen, psnr_opt, l1_loss, mse_loss,
              g_scalar, psnr_writer, psnr_step):
    
    gen.train()
    total_psnr = 0.0
    loop = tqdm(loader, leave = True, position = 0) 
    for idx, (hr, lr) in enumerate(loop):
        
        hr = hr.to(config.DEVICE)
        lr = lr.to(config.DEVICE)
        
        psnr_opt.zero_grad()
        with torch.cuda.amp.autocast():
            
            sr = gen(lr)
            loss = l1_loss(sr, hr)
            total_psnr += 10 * torch.log10(1.0 / mse_loss(sr, hr))

        g_scalar.scale(loss).backward()
        g_scalar.step(psnr_opt)
        g_scalar.update()
        
        psnr = total_psnr/len(loader)
        psnr_writer.add_scalar('PSNR Train/PSNR', psnr, psnr_step)
        psnr_writer.add_scalar('PSNR Train/Pixel_loss', loss.item(), psnr_step)
        psnr_step += 1

    print("\nPSNR: ", psnr)
    return psnr_step
      
   
def train(loader, gen, disc, g_opt, d_opt, l1_loss, content_loss, 
          mse_loss, g_scalar, d_scalar, writer, step):
    
    
    loop = tqdm(loader, leave = True, position = 0)
    total_psnr = 0.0
    for idx, (hr, lr) in enumerate(loop):
        
        gen.train()
        hr = hr.to(config.DEVICE)
        lr = lr.to(config.DEVICE)
        
        with torch.cuda.amp.autocast():
            
            fake = gen(lr)
            disc_real = disc(hr)
            disc_fake = disc(fake.detach())
            gp = gradient_penalty(disc, hr, fake)
            disc_loss = (-(torch.mean(disc_real) - torch.mean(disc_fake)) + config.LAMBDA_GP * gp)
            
        d_opt.zero_grad()
        d_scalar.scale(disc_loss).backward()
        d_scalar.step(d_opt)
        d_scalar.update()
        
        with torch.cuda.amp.autocast():
            
            pixel_loss = l1_loss(fake, hr)
            total_psnr += 10 * torch.log10(1.0 / mse_loss(fake, hr))
            cont_loss = content_loss(fake, hr)
            gan_loss = (-torch.mean(disc(fake)))
            gen_loss = 0.01 * pixel_loss + cont_loss + 0.005 * gan_loss
            
        g_opt.zero_grad()
        g_scalar.scale(gen_loss).backward()
        g_scalar.step(g_opt)
        g_scalar.update()
        
        gen.eval()
        psnr = total_psnr/len(loader)
        writer.add_scalar('PSNR/PSNR Score', psnr, global_step = step)
        writer.add_scalar("disc_loss/Discriminator loss", disc_loss.item(), global_step = step)
        writer.add_scalar("Gen_losses/Generator loss", gen_loss.item(), global_step = step)
        writer.add_scalar("Gen_losses/Pixel loss", pixel_loss.item(), global_step = step)
        writer.add_scalar("Gen_losses/Content loss", cont_loss.item(), global_step = step)
        writer.add_scalar("Gen_losses/Adversarial loss", gan_loss.item(), global_step = step)
        
        step += 1

    return step

    
def main():
    
    dataset = TrainDataset(config.TRAIN_DATA_PATH)
    loader = DataLoader(dataset, batch_size = config.BATCH_SIZE, shuffle = True,
                        pin_memory = True, num_workers = 2)
    
    gen = Generator(in_c = 3, out_c = 64).to(config.device)
    disc = Discriminator().to(config.device)
    init_weights(gen)

    fixed_lr = T.Compose([T.Resize((32, 32), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])(Image.open(config.FIXED_LR_PATH)).unsqueeze(0)
    fixed_lr = fixed_lr.to(config.DEVICE)
    fixed_lr1 = T.Compose([T.Resize((32, 32), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])(Image.open(config.FIXED_LR1_PATH)).unsqueeze(0)
    fixed_lr1 = fixed_lr1.to(config.DEVICE)

    psnr_opt = torch.optim.Adam(gen.parameters(), lr = config.PSNR_OPT_LR, betas = (0.9, 0.999))
    psnr_writer = SummaryWriter(config.PSNR_LOGS_PATH)
    psnr_step = 0

    writer = SummaryWriter(config.GAN_LOGS_PATH)
    step = 0

    g_opt = torch.optim.Adam(gen.parameters(), lr = config.LR, betas = (0.9, 0.999))
    d_opt = torch.optim.Adam(disc.parameters(), lr = config.LR, betas = (0.9, 0.999))
    
    g_scalar = torch.cuda.amp.GradScaler()
    d_scalar = torch.cuda.amp.GradScaler()
    
    l1_loss = nn.L1Loss()
    content_loss = ContentLoss()
    mse_loss = nn.MSELoss()
    
    gen.train()
    disc.train()

    print("Starting PSNR Training...")
    #Psnr Training
    for epoch in range(100):

        print(f"Epoch: {epoch} \n")
        psnr_step = psnrTrain(loader, gen, psnr_opt, l1_loss, mse_loss, g_scalar, psnr_writer, psnr_step)
        
        torch.save(gen.state_dict(), config.PSNR_GEN_PATH)

        gen.eval()
        with torch.no_grad():
          sr = gen(fixed_lr)
          sr1 = gen(fixed_lr1)
          psnr_writer.add_image('PSNR/real', fixed_lr.squeeze(0).cpu(), global_step = psnr_step)
          psnr_writer.add_image('PSNR/fake', sr.squeeze(0).cpu(), global_step = psnr_step)
          psnr_writer.add_image('PSNR/real1', fixed_lr1.squeeze(0).cpu(), global_step = psnr_step)
          psnr_writer.add_image('PSNR/fake1', sr1.squeeze(0).cpu(), global_step = psnr_step)
    
    gen.load_state_dict(torch.load(config.PSNR_GEN_PATH))
    #GAN Training
    print("Starting GAN Training...")
    for epoch in range(config.NUM_EPOCHS):
        
        print(f"Epoch: {epoch} \n")
        step = train(loader, gen, disc, g_opt, d_opt, l1_loss, content_loss, 
                     mse_loss, g_scalar, d_scalar, writer, step)
        with torch.no_grad():
            
            gen_img = gen(fixed_lr)
            gen_img1 = gen(fixed_lr1)
            vutils.save_image(gen_img.detach(), open(config.SAVE_IMG1_PATH + f'img{epoch}.png', 'wb'), normalize = True)
            vutils.save_image(gen_img1.detach(), open(config.SAVE_IMG2_PATH + f'img{epoch}.png', 'wb'), normalize = True)
            gen.train()

        writer.add_image("Real/Image1", fixed_lr.squeeze(0), global_step = step)
        writer.add_image("Real/Image2", fixed_lr1.squeeze(0), global_step = step)
        writer.add_image("Generated Images/image1", gen_img.squeeze(0), global_step = step)
        writer.add_image("Generated Images/image2", gen_img1.squeeze(0), global_step = step)    
        torch.save(gen.state_dict(), config.GEN_PATH)
        torch.save(disc.state_dict(), config.DISC_PATH) 

if __name__ == '__main__':

    main()  