import torch
import torch.nn as nn
from torchsummary import summary

def init_weights(model, scale = 0.1):
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale
    

class ConvBlock(nn.Module):
    
    def __init__(self, in_c, out_c, use_lrelu, **kwargs):
        
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, **kwargs, bias = True)
        self.lrelu = nn.LeakyReLU(0.2, inplace = True) if use_lrelu else nn.Identity()
        
    def forward(self, x):
        
        return self.lrelu(self.conv(x))
    
    
class DenseBlock(nn.Module):
    
    def __init__(self, in_c, hidden_c, beta = 0.2):
        
        super().__init__()
        self.beta = beta
        self.denseblocks = nn.ModuleList()
        
        for i in range(5):
          
            self.denseblocks.append(ConvBlock(in_c + hidden_c * i,
                                              hidden_c if i <= 3 else in_c ,
                                              use_lrelu = True if i <= 3 else False,
                                              kernel_size = 3,
                                              stride = 1,
                                              padding = 1
                                              )
                                    )
    def forward(self, x):
        
        inputs = x
        for block in self.denseblocks:
            res = block(inputs)
            inputs = torch.cat([inputs, res], dim = 1)
            
        return x + res * self.beta
  
    
class RRDB(nn.Module):
    
    def __init__(self, in_c, beta = 0.2):
        
        super().__init__()
        
        self.beta = beta
        self.rrdbs = nn.Sequential(*[DenseBlock(in_c, hidden_c = 32) for i in range(3)])
        
    def forward(self, x):
        
        return x + self.beta * self.rrdbs(x)
  
    
class Upsampling(nn.Module):
    
    def __init__(self, in_c, scale_factor = 2):
        
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = scale_factor, mode = 'bilinear')
        self.conv = nn.Conv2d(in_c, in_c, 3, 1, 1, bias = False)
        self.lrelu = nn.LeakyReLU(0.2, inplace = True)
        
    def forward(self, x):
        
        return self.lrelu(self.conv(self.upsample(x)))
  
    
class Generator(nn.Module):
    
    def __init__(self, in_c, out_c, num_blocks = 23):
        
        super().__init__()
        self.first_layer = nn.Conv2d(in_c, out_c, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.res = nn.Sequential(*[RRDB(out_c) for _ in range(num_blocks)])
        self.conv_before_upsample = nn.Conv2d(out_c, out_c, kernel_size = 3, stride = 1, padding = 1)
        self.upsample = nn.Sequential(Upsampling(out_c), Upsampling(out_c))
        self.conv_after_upsample = nn.Sequential(nn.Conv2d(out_c, out_c, kernel_size = 3, stride = 1, padding = 1, bias = True),
                                                 nn.LeakyReLU(0.2, inplace = True),
                                                 nn.Conv2d(out_c, in_c, kernel_size = 3, stride = 1, padding = 1, bias = True))
        
    def forward(self, x):
        
        first = self.first_layer(x)
        x = self.conv_before_upsample(self.res(first)) + first
        x = self.conv_after_upsample(self.upsample(x))
        return x


class Discriminator(nn.Module):
    
    def __init__(self, in_c = 3, out_channels = [64, 64, 128, 128, 256, 256, 512, 512]):
        
        super().__init__()
        layers = []
        
        for i, hidden_c in enumerate(out_channels):
            
            layers.append(ConvBlock(in_c, out_c = hidden_c, use_lrelu = True, 
                                    kernel_size = 3, stride = 1 + i % 2, padding = 1))
            in_c = hidden_c
            
        self.layers = nn.Sequential(*layers)
        self.last_layers = nn.Sequential(nn.AdaptiveAvgPool2d((6, 6)),
                                         nn.Flatten(),
                                         nn.Linear(out_channels[-1] * 6 * 6, 1024),
                                         nn.LeakyReLU(0.2, inplace = True),
                                         nn.Linear(1024, 1)
                                         )
    
    def forward(self, x):
        
        out = self.layers(x)
        out = self.last_layers(out)
        return out


if __name__ == '__main__':
    
    x = torch.randn((2, 3, 32, 32))
    gen = Generator(in_c = 3, out_c = 64)
    disc = Discriminator()
    
    print(summary(gen, (3, 32, 32)))
    print(summary(disc, (3, 128, 128)))
    
    g_output = gen(x)
    d_output = disc(g_output)
    print(f"\nGenerator Output Shape: {g_output.shape}\n Discriminator Output Shape: {d_output.shape}")