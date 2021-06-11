import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from dcgan import Generator

Z_DIM = 100
PATH = 'E:\Pytorch\DCGAN\generator_at_epoch10.pth'

model = torch.load(PATH)
netG = Generator(Z_DIM,64)
netG.load_state_dict(model)

noise = torch.randn(64,Z_DIM,1,1)

with torch.no_grad():
  gen_img = netG(noise).detach().cpu()
  
plt.figure(figsize = (16,16))
plt.axis('off')
plt.title('Generated Images')
plt.imshow(np.transpose(vutils.make_grid(gen_img,padding = 2, normalize = True), (1,2,0)))
plt.show()