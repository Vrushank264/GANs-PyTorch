# Progressive Growing of GANs (PROGAN)

Pytorch implementation of [ProGAN](https://arxiv.org/abs/1710.10196)

- [ ] add a blog 

## Progressive Growing: 
`Transition(4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128)`
<p align = 'center'>
    <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ProGAN/Progressive%20Growing/Training/4x4.png">
</p>
<p align = 'center'>
    :arrow_down:
</p>
<p align = 'center'>
    <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ProGAN/Progressive%20Growing/Training/8x8.png">
</p>
<p align = 'center'>
    :arrow_down:
</p>
<p align = 'center'>
    <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ProGAN/Progressive%20Growing/Training/16x16.png">
</p>
<p align = 'center'>
    :arrow_down:
</p>
<p align = 'center'>
    <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ProGAN/Progressive%20Growing/Training/32x32.png">
</p>
<p align = 'center'>
    :arrow_down:
</p>
<p align = 'center'>
    <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ProGAN/Progressive%20Growing/Training/64x64.png">
</p>
<p align = 'center'>
    :arrow_down:
</p>
<p align = 'center'>
    <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ProGAN/Progressive%20Growing/Training/128x128.png">
</p>


## Results:

Interpolations:

1st one changes the gender while the other two change the hair colour. [You can see more in `Generated Images/Interpolations`]

<p float="left">
  <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ProGAN/Generated%20Images/Interpolations/interpolation_woman2man.gif" width="200" />
  <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ProGAN/Generated%20Images/Interpolations/interpolation_brown2blone.gif" width="200" /> 
  <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ProGAN/Generated%20Images/Interpolations/interpolation_blonde2brown.gif" width="200" />
</p>

The model often produces really good images. :smiley:
<p float="left">
  <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ProGAN/Generated%20Images/img1.png" width="150" />
  <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ProGAN/Generated%20Images/img4.png" width="150" /> 
  <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ProGAN/Generated%20Images/img3.png" width="150" />
  <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ProGAN/Generated%20Images/img6.png" width="150" />
  <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ProGAN/Generated%20Images/img7.png" width="150" />  
</p>


But, sometimes it doesn't... :stuck_out_tongue_closed_eyes:
<p float="left">
  <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ProGAN/Generated%20Images/img2.png" width="150" />
  <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/ProGAN/Generated%20Images/img5.png" width="150" /> 
</p>

> Details:

- For this experiment, I used [CelebA-HQ Dataset](https://www.kaggle.com/lamsimon/celebahq).
- Most of the hyperparameters are set according to the paper, but you can modify them into the `config.py` file.
- I trained the model until image size 128x128. Model was trained for 20 epochs for every image resolution.

- If you want to train this model on your own dataset, then,
    1. Clone this repo. 
    2. Specify root folder and other hyperparameters in `config.py` file.
    3. Run `train.py` file.

> Interpolation:

- By interpolating in high dimensional latent space, We can know how neural networks represent data. Neural networks model similar data points close to each other, so if we interpolate from datapoint A to datapoint B, we can see a particular image changing as it moves from A to B.

- This experiment uses a technique called [`slerp`](https://arxiv.org/pdf/1609.04468.pdf) to interpolate.
- You can create your own interpolation gif by,
1) Clone this repo.
2) Download pretrained generator model from `Pretrained Models` folder.
3) Specify **gen_path** (path to the generator) and **save_dir** in `config.py` file.
4) Run `interpolate.py`.

