import config
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
from PIL import Image
from models import Generator

IMG_PATH = '/content/test/0898.png' 

gen = Generator(3, 64).to(config.DEVICE)
gen.load_state_dict(torch.load(config.GEN_PATH))
print("Model Loaded!")

fixed_lr = T.ToTensor()(Image.open(IMG_PATH).convert('RGB'))
plt.imshow(np.transpose(fixed_lr, (1, 2, 0)))
fixed_lr = fixed_lr.unsqueeze(0).to(config.DEVICE)

gen.eval()
with torch.no_grad():

  gen_img = gen(fixed_lr)
  print(gen_img.shape)
  plt.imshow(np.transpose(gen_img.squeeze(0).cpu() , (1, 2, 0)))