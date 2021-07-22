import torch
import numpy as np
import os
from matplotlib.pyplot import imsave
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import imageio
from Models import Generator


def slerp(val, low, high):
    
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    sin_omega = np.sin(omega)
    return np.sin((1.0 - val) * omega) / sin_omega * low + np.sin(val * omega) / sin_omega * high

def slerp_interpolation(num_latents, num_interpolations):
    
    low = np.random.randn(256)
    latent_interps = np.empty(shape = (0, 256), dtype = np.float32)
    
    for _ in range(num_latents):
        
        high = np.random.randn(256)
        interp_vals = np.linspace(1. / num_interpolations, 1, num = num_interpolations)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals], dtype = np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))
        low = high
    
    return latent_interps[:, :, np.newaxis, np.newaxis]

def scale_image(img, drange_in, drange_out):
    
    scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
    bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
    image = np.clip(img * scale + bias, 0, 255).astype(np.uint8)
    return image

def save_images(images, output_dir, start_idx=0):
    for i, image in enumerate(images):
        image = scale_image(image, [-1, 1], [0, 255])
        image = image.transpose(1, 2, 0) 
        image_path = os.path.join(output_dir,'image{:04d}.png'.format(i+start_idx))
        imsave(image_path, image)

def create_gif(png_dir):
    
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
                file_path = os.path.join(png_dir, file_name)
                images.append(imageio.imread(file_path))
    imageio.mimsave(os.path.join(png_dir, 'interpolation.gif'), images)

class LatentDataset(Dataset):
    
    def __init__(self, num_latents = 1, filter_latents = 3, num_interpolations = 50):
        
        latents = slerp_interpolation(num_latents, num_interpolations)
        self.data = torch.from_numpy(latents)
        
    def __getitem__(self, index):
        
        return self.data[index]
    
    def __len__(self):
        
        return len(self.data)
    
def main():

    gen_path = 'E:/Computer Vision/ProGAN/Saved Models/Generator_128_final.pth'
    save_dir ='E:/Computer Vision/ProGAN/Image/interpolation'
    
    np.random.seed(76)
    torch.manual_seed(76)
    print('Loading Generator...')
    model = Generator(256,256)
    model.load_state_dict(torch.load(gen_path, map_location = torch.device('cpu')))
    
    latent_dataset = LatentDataset()
    loader = DataLoader(latent_dataset, 100, shuffle = False)
    print('Processing...')
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, position = 0, leave = True)):
            
            output = model(data, 0.99, 5)
            imgs_np = output.data.numpy()
            save_images(imgs_np, output_dir = save_dir)
            
    create_gif(save_dir)
    
if __name__ == '__main__':
    
    main()
    
