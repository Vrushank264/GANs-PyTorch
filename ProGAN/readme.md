# Progressive Growing of GANs (PROGAN)

Pytorch implementation of [ProGAN]()

[x] add a blog 

## Results:

Interpolations:

![]
![]



The model often produces really good images.
![]
![]

But, sometimes it doesn't...
![]
![]

> Details:

- For this experiment, I used [CelebA-HQ Dataset]().
- Most of the hyperparameters are set according to the paper, but you can modify them into the `config.py` file.
- I trained the model until image size 128x128. Model was trained for 20 epochs for every resolution.
(4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128)
- If you want to train this model on your own dataset, then,
    1. Clone this repo. 
    2. Specify root folder and other hyperparameters in `config.py` file.
    3. Run `train.py` file.

> Interpolation:

- By interpolating in high dimensional latent space, We can know how neural networks represent data. Neural networks model similar data points close to each other, so if we interpolate from datapoint A to datapoint B, we can see a particular image changing as it moves from A to B. [See examples below]

- This experiment uses a technique called [`slerp`](https://arxiv.org/pdf/1609.04468.pdf) to interpolate.
- You can create your own interpolation gif by,
1) Clone this repo.
2) Download pretrained generator model from `Trained Model` folder.
3) Specify **gen_path** (path to the generator) and **save_dir** in `config.py` file.
4) Run `interpolate.py`.

