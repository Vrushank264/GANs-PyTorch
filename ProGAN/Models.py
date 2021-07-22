import torch
import torch.nn as nn
import torch.nn.functional as fun
from math import log2

factor = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


class ProConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, gain=2):
        super(ProConv, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.eqlr = (gain / (in_c * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        out = self.conv(x * self.eqlr)
        return out + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNormalization(nn.Module):
    def __init__(self):
        super(PixelNormalization, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = ProConv(in_c, out_c)
        self.conv2 = ProConv(out_c, out_c)
        self.lrelu = nn.LeakyReLU(0.2)
        self.pixelnorm = PixelNormalization()

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.pixelnorm(x) if self.use_pn else x
        x = self.lrelu(self.conv2(x))
        x = self.pixelnorm(x) if self.use_pn else x
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim, in_c, img_c=3):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            PixelNormalization(),
            nn.ConvTranspose2d(latent_dim, in_c, 4, 1, 0),
            nn.LeakyReLU(0.2),
            ProConv(in_c, in_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNormalization(),
        )

        self.initial_rgb = ProConv(in_c, img_c, kernel_size=1, stride=1, padding=0)
        self.Progressive_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb]),
        )

        for i in range(len(factor) - 1):
            conv_in_c = int(in_c * factor[i])
            conv_out_c = int(in_c * factor[i + 1])
            self.Progressive_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(
                ProConv(conv_out_c, img_c, kernel_size=1, stride=1, padding=0)
            )

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):
        out = self.initial(x)
        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = fun.interpolate(out, scale_factor=2, mode="nearest")
            out = self.Progressive_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)

        return self.fade_in(alpha, final_upscaled, final_out)


class Discriminator(nn.Module):
    def __init__(self, in_c, img_c=3):
        super().__init__()
        self.progressive_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.lrelu = nn.LeakyReLU(0.2)

        for i in range(len(factor) - 1, 0, -1):
            conv_in_c = int(in_c * factor[i])
            conv_out_c = int(in_c * factor[i - 1])
            self.progressive_blocks.append(
                ConvBlock(conv_in_c, conv_out_c, use_pixelnorm=False)
            )
            self.rgb_layers.append(
                ProConv(img_c, conv_in_c, kernel_size=1, stride=1, padding=0)
            )

        self.initial_rgb = ProConv(img_c, in_c, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.rgb_layers.append(self.initial_rgb)

        self.final_block = nn.Sequential(
            ProConv(in_c + 1, in_c, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            ProConv(in_c, in_c, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            ProConv(in_c, 1, kernel_size=1, stride=1, padding=0),
        )

    def fade_in(self, alpha, out, downscaled):
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_Std(self, x):
        batch_stats = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_stats], dim=1)

    def forward(self, x, alpha, steps):
        cur_step = len(self.progressive_blocks) - steps
        out = self.lrelu(self.rgb_layers[cur_step](x))
        if steps == 0:
            out = self.minibatch_Std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.lrelu(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.progressive_blocks[cur_step](out))
        out = self.fade_in(alpha, out, downscaled)

        for step in range(cur_step + 1, len(self.progressive_blocks)):
            out = self.progressive_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_Std(out)
        return self.final_block(out).view(out.shape[0], -1)


if __name__ == "__main__":

    latent_dim = 50
    in_c = 256
    gen = Generator(latent_dim, in_c, img_c=3)
    disc = Discriminator(in_c, img_c=3)

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:

        num_steps = int(log2(img_size / 4))
        x = torch.randn((1, latent_dim, 1, 1))
        z = gen(x, 0.5, num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = disc(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success! at image size: {img_size}")
