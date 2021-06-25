import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda')
lr = 2e-4
batch_size = 4
num_worker = 2
img_size = 256
l1_lambda = 100
num_epochs = 50
root_dir = 'E:/Computer Vision/Pix2pix/sketch2Anime/train'
save_dir = 'E:/Computer Vision/Pix2pix/Results'

both_transform = A.Compose(
    [A.Resize(width = 256, height = 256)],
    additional_targets = {'image0':'image'}
    )

transform_only_input = A.Compose(
    [A.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5], max_pixel_value = 255.0),
     ToTensorV2()
     ]
    )

transform_only_mask = A.Compose(
    [
     A.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5], max_pixel_value = 255.0),
     ToTensorV2()
     ]
    )
