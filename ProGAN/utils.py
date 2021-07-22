import torch
import torchvision.utils as vutils
from scipy.stats import truncnorm
import config


def plot_tensorboard(writer, d_loss, g_loss, real, fake, step):

    writer.add_scalar("Discriminator Loss; ", d_loss, global_step=step)
    writer.add_scalar("Generator Loss: ", g_loss, global_step=step)
    with torch.no_grad():
        img_grid_real = vutils.make_grid(real[:8], normalize=True)
        img_grid_fake = vutils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=step)
        writer.add_image("Fake", img_grid_fake, global_step=step)


def gradient_penalty(disc, real, fake, alpha, train_step, device=torch.device("cuda")):

    batch_size, c, h, w = real.shape
    beta = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(device)
    interpolated_imgs = real * beta + fake.detach() * (1 - beta)
    interpolated_imgs.requires_grad_(True)

    mixed_scores = disc(interpolated_imgs, alpha, train_step)
    gradient = torch.autograd.grad(
        inputs=interpolated_imgs,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    grad_norm = gradient.norm(2, dim=1)
    grad_penalty = torch.mean((grad_norm - 1) ** 2)
    return grad_penalty


def save_img(gen, steps, truncation=0.7, n=64):
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.tensor(
                truncnorm.rvs(
                    -truncation, truncation, size=(1, config.latent_dim, 1, 1)
                ),
                device=config.device,
                dtype=torch.float32,
            )
            img = gen(noise, alpha, steps)
            vutils.save_image(img * 0.5 + 0.5, f"E:/Computer Vision/ProGAN/generated_images/img{i}.png")
    gen.train()
