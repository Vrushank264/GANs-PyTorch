# Deep Convolutional GAN (DCGAN)

 Pytorch Implementation of [DCGAN](https://arxiv.org/abs/1511.06434.pdf)

- A Generative Adversarial Network(GAN) contains 2 distinct Neural Networks:
    1) Generator
    2) Discriminator

- Generator takes in random noise as an input and generates an image, then that image is sent to the Discriminator network which identifies whether it is fake or real. Then adverserial loss is computed and then backpropagated to further improve the performance of both the networks.

![DCGAN architecture](https://github.com/Vrushank264/GANs-PyTorch/blob/main/DCGAN/dcgan_architecture.png)

> Some Important points to keep in mind:
- Use Batch Normalization in both the networks.
- Don't use pooling layers, instead use strided convolutions.
- Remove fully connected hidden layers.
- Use ReLU activation in all generator layers except the output layer, use tanh for that.
- Use LeakyReLU activation in all Discriminator layers except the output layer, use sigmoid for that.
- Don't forget to initialize the weights!


> Details:
- I use [Zalando Fashion image dataset](https://www.kaggle.com/dqmonn/zalando-store-crawl) and resize it to 64x64.
- Both the networks are quite deep in the official paper, but i use relatively less layers and features. 
- learning rates, optimizer and other hyperparameters are set according to the official paper.
- Model is trained for only 10 epochs, here are the results:

> If you want to generate some images, then download the pretrained model provided in ``Trained model`` folder, then specify the path into the ``img_generator.py`` file and run it.

<tr>
  <td>
Training Images:  
<img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/DCGAN/Generated%20Images/Training%20Image.png" width="350" height="400">
  </td>
  <td>
    Random Noise
<img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/DCGAN/Generated%20Images/random_noise.png" width="350" height="400">
  </td>
</tr>
<tr>
  <td>
  Epoch 1:
  <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/DCGAN/Generated%20Images/epoch1.png" width="350" height="400">
  </td>
  <td>
    Epoch 10:
    <img src="https://github.com/Vrushank264/GANs-PyTorch/blob/main/DCGAN/Generated%20Images/epoch10.png" width="350" height="400">
  </td>
 </tr>
 
 
