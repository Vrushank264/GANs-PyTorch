# Pix2Pix

Pytorch implementation of [Pix2Pix](https://arxiv.org/abs/1611.07004)

- In my opinion, Pix2Pix revolutionalized the image to image translation.
-  It is a type of conditional generative adversarial network (cGAN) that learns a mapping from input images to output images.

### Architecture:

![Architecture](https://github.com/Vrushank264/GANs-PyTorch/blob/main/Pix2pix/pix2pix_architecture.jpg)

>Generator:
- A generator has a U-Net-based architecture.
- Each block in the encoder is: Convolution -> Batch normalization -> Leaky ReLU if parameter ``down`` in ``Block`` function is `True`
- Each block in the decoder is: Transposed convolution -> Batch normalization -> Dropout(Only in first 3 layers) -> ReLU if parameter ``down`` in ``Block`` function is `False`
- There are `skip connections` between the encoder and decoder.

>Discriminator:

- A discriminator is a convolutional PatchGAN classifier (proposed in the paper).
- Use `reflect` as padding_mode in every convolutions.
- Each CNNBlock in the discriminator is: Convolution -> Batch normalization -> Leaky ReLU.
- The discriminator receives 2 inputs:
    1. The input image and the target image, which it should classify as real.
    2. The input image and the generated image (the output of the generator), which it should classify as fake.

> Details:

- For this experiment, I used [sketch2Anime Dataset](https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair).
- Most of the hyperparameters are set according to the paper, but you can modify them into the `config.py` file.
- To generate Images of your own, you have to,
    1. Download the [sketch2Anime Dataset](https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair).
    2. Download pretrained Generator model from `Trained Models` folder. (which is on google drive)
    3. Change val_dir and gen_path location in config file.
    4. Run `imgGenerator.py` script.
- I ran it for 50 epochs, and here is the result. (You can see more of them in the `Results` folder)

Input Sketch:
![Input sketch](https://github.com/Vrushank264/GANs-PyTorch/blob/main/Pix2pix/Results/input_2.png)

Generated Image:
![Generated Image](https://github.com/Vrushank264/GANs-PyTorch/blob/main/Pix2pix/Results/generated_2.png)

Target:
![Target](https://github.com/Vrushank264/GANs-PyTorch/blob/main/Pix2pix/Results/label_2.png)
