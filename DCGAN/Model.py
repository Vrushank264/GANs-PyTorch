import torch
import torch.nn as nn
import torch.nn.functional as fun

def weights_gen(w):
    
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data,0.0,0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data,1.0,0.02)
        nn.init.constant_(w.bias.data,0)
        
#Generator Architecture
class Generator(nn.Module):
    def __init__(self,z_dim,features_g):
        super(Generator,self).__init__()
        
        self.tconv1 = nn.ConvTranspose2d(z_dim, features_g*8, kernel_size = 4, stride = 1, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(features_g*8)
        
        self.tconv2 = nn.ConvTranspose2d(features_g*8, features_g*4, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(features_g*4)
        
        self.tconv3 = nn.ConvTranspose2d(features_g*4, features_g*2, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(features_g*2)
        
        self.tconv4 = nn.ConvTranspose2d(features_g*2, features_g, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.bn4 = nn.BatchNorm2d(features_g)
        
        self.tconv5 = nn.ConvTranspose2d(features_g, 3,4,2,1,bias = False)
        
    def forward(self, x):
        
        x = fun.relu(self.bn1(self.tconv1(x)))
        x = fun.relu(self.bn2(self.tconv2(x)))
        x = fun.relu(self.bn3(self.tconv3(x)))
        x = fun.relu(self.bn4(self.tconv4(x)))
        x = fun.tanh(self.tconv5(x))

        return x

#Discriminator Architecture    
class Discriminator(nn.Module):
    
    def __init__(self,features_d):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(3,features_d, 4, 2, 1, bias = False)
        
        self.conv2 = nn.Conv2d(features_d, features_d*2, 4, 2, 1, bias = False)
        self.bn2 = nn.BatchNorm2d(features_d*2)
        
        self.conv3 = nn.Conv2d(features_d*2, features_d*4, 4, 2, 1, bias = False)
        self.bn3 = nn.BatchNorm2d(features_d*4)
        
        self.conv4 = nn.Conv2d(features_d*4, features_d*8, 4, 2, 1, bias = False)
        self.bn4 = nn.BatchNorm2d(features_d*8)
        
        self.conv5 = nn.Conv2d(features_d*8, 1, 4, 1, 0,bias = False)
        
    def forward(self, x):
        
        x = fun.leaky_relu(self.conv1(x), 0.2, True)
        x = fun.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = fun.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = fun.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = fun.sigmoid(self.conv5(x))
        
        return x
    