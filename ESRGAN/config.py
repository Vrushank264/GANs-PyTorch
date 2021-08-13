import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-4
PSNR_OPT_LR = 2e-4 
LAMBDA_GP = 10

TRAIN_DATA_PATH = '/content/train'
FIXED_LR_PATH = '/content/test/0822.png'
FIXED_LR1_PATH = '/content/test/0855.png'
PSNR_GEN_PATH = '/content/drive/MyDrive/Esrgan/PSNR_Generator.pth'    
PSNR_LOGS_PATH = '/content/drive/MyDrive/Esrgan/psnr'
GAN_LOGS_PATH = '/content/drive/MyDrive/Esrgan/logs'   
GEN_PATH = '/content/drive/MyDrive/Esrgan/Generator.pth'
DISC_PATH = '/content/drive/MyDrive/Esrgan/Discriminator.pth'
SAVE_IMG1_PATH = '/content/drive/MyDrive/Esrgan/img2'
SAVE_IMG2_PATH = '/content/drive/MyDrive/Esrgan/img1'
