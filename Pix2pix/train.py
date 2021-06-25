import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from generator import Generator
from discriminator import Discriminator
from utils import save_img
import config
from dataset import MapDataset
import datetime as dt

i = 1
D_l = []
G_l = []

def train(disc, gen, loader, opt_d, opt_g, l1, bce, g_scalar, d_scalar):

    loop = tqdm(loader, leave=True, position = 0)
    for idx, (x,y) in enumerate(loop):
        
        x,y = x.to(config.device), y.to(config.device)
        
        #-----------------------------------
        #   Discriminator(disc) Training
        #-----------------------------------
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x,y)
            D_fake = disc(x, y_fake.detach())
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
            
        opt_d.zero_grad()
        d_scalar.scale(D_loss).backward()
        d_scalar.step(opt_d)
        d_scalar.update()
        
        #----------------------------------
        #    Generator(gen) Training
        #----------------------------------
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, y) * config.l1_lambda
            G_loss = G_fake_loss + L1
          
        opt_g.zero_grad()
        g_scalar.scale(G_loss).backward()
        g_scalar.step(opt_g)
        g_scalar.update()
        G_l.append(G_loss.item())
        D_l.append(D_loss.item())
        
        #-----------------------
        #  Plotting the losses
        #-----------------------
        if idx % 400 == 0: 
            plt.figure(figsize = (10,5))
            plt.title('Generator and Discriminator loss')
            plt.plot(G_l,label = 'Generator')
            plt.plot(D_l,label = 'Discriminator')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            print('Minimum G_L: ', min(G_l))
            print('Minimum D_L: ', min(D_l))
            save_img(gen, loader, idx, folder = config.save_dir)
            
        

def main():
    
    #model_g = torch.load('E:/Computer Vision/Pix2pix/Generator.pth')
    #model_d = torch.load('E:/Computer Vision/Pix2pix/Discriminator.pth')
    disc = Discriminator().to(config.device)
    gen = Generator().to(config.device)
    #gen.load_state_dict(model_g)
    #disc.load_state_dict(model_d)
    opt_d = optim.Adam(disc.parameters(), lr = config.lr, betas = (0.5,0.999))
    opt_g = optim.Adam(gen.parameters(), lr = config.lr, betas = (0.5,0.999))
    
    bce = nn.BCEWithLogitsLoss()
    L1_loss = nn.L1Loss()
    
    train_dataset = MapDataset(root_dir = config.root_dir)
    train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True, num_workers=0)
    
    g_scalar = torch.cuda.amp.GradScaler() 
    d_scalar = torch.cuda.amp.GradScaler()
    
    val_dataset = MapDataset(root_dir = config.val_dir)
    val_loader = DataLoader(val_dataset, batch_size = 8, shuffle = True)
    
    for epoch in range(config.num_epochs):
        print("Epoch: ", epoch)
        train(disc, gen, train_loader,opt_d, opt_g, L1_loss, bce, g_scalar, d_scalar)
        
        if epoch % 5 == 0:
            torch.save(gen.state_dict(), open(config.root_dir + '/Generator.pth', 'wb'))
            torch.save(disc.state_dict(), open(config.root_dir + '/Discriminator.pth', 'wb'))
            save_img(gen, val_loader, epoch, folder = config.save_dir)
            
        save_img(gen, val_loader, epoch, folder = config.save_dir)

if __name__ == '__main__':
    
    main()
