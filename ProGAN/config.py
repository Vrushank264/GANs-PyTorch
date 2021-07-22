import torch

root = "E:/Computer Vision/ProGAN/celeba_hq"
save_path = "E:/Computer Vision/ProGAN/Saved Models"
device = torch.device('cuda')
#print(torch.cuda.is_available())
lr = 1e-3
batch_sizes = [32, 32, 32, 8, 16, 8, 4, 1]
img_c = 3
latent_dim = 256
in_c = 256
lambda_gp = 10
progressive_epochs = [15] * len(batch_sizes)
fixed_noise = torch.randn((8, latent_dim, 1, 1)).to(torch.device('cuda'))
