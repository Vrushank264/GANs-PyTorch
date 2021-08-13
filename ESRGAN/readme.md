# Enhanced Super-Resolution Generative Adversarial Network (ESRGAN)

Pytorch implementation of [ESRGAN](https://arxiv.org/abs/1809.00219)


## Results:

1) Input: 64x64 image and Output: 256x256 image
<p float="left" align = 'center'>
  <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ESRGAN/Generated/LR_Input_64x64.png" width = 256/>
  :arrow_right:
  <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ESRGAN/Generated/SR_Output_256x256.png"/> 
</p>

2) Input: 128x128 image and Output: 512x512 image
<p align = 'center'>
  <img src= "https://github.com/Vrushank264/GANs-PyTorch/blob/main/ESRGAN/Results/128_to_512.png" />
</p>


3) Input: 256x256 image and Output: 1024x1024 image
<p align = 'center'>
  <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ESRGAN/Generated/Low%20Res%20Inputs(256x256)/LR.png"/>
</p>
<p align = 'center'>  
  ⬇️:
</p>
<p align = 'center'>
  <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ESRGAN/Generated/Super%20Res%20Outputs(1024x1024)/SR.png"/> 
</p>

More Examples are in `Results` and `Generated` folders.

### Details:

- This model is trained only on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset with a task to convert 32x32 images to 128x128 images. After training, model can generalize on any Image resolution, and upscales the input image by the factor of 4.
- The generator is first trained to maximize the PSNR, and minimize the pixel loss / L1 loss.
- Then the weights of PSNR generator are used to initialize the GAN generator for the adversarial training which minimizes 3 losses in total namely L1 loss, VGG Loss / Content Loss, and GAN loss.
- If you want to train your own model, then, 
1) Run `data/get_div2k_data.sh` to download train and test data.
2) specify necessary paths and hyperparameters in `config.py` file.
3) Run `train_esrgan.py` file.

### TODOs:

1) Train model on more data in order to capture and model wide range of patterns perfectly.

