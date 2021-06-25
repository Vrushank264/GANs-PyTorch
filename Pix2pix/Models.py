# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:17:43 2021

@author: Admin
"""


import torch
import torch.nn as nn
import torch.nn.functional as fun
from math import log2

class ProConv(nn.Module):
    
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_c,out_c,kernel_size,stride,padding)
        self.eqlr = (gain / (in_c * kernel_size * kernel_size))**0.5
        self.bias = self.conv.bias
        self.conv.bias = None
        
        nn.init.normal_(self.conv.weights)
        nn.init.zeros_(self.bias)
        
        def forward(self, x):
            return (self.conv(x * self.eqlr)) + self.bias.view(1, self.bias.shape[0], 1, 1)
        
class PixelNormalization(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8
        
        def forward(self, x):
            return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim = True) + self.epsilon)
        
class ConvBlock(nn.Module):
    
    def __init__(self, in_c, out_c, use_pixelnorm = True):
        super().__init__()
        self.conv1 = ProConv(in_c, out_c)
        self.conv2 = ProConv(out_c, out_c)
        self.lrelu = nn.LeakyReLU(0.2)
        self.pixelnorm = PixelNormalization()
        self.use_pn = use_pixelnorm
        
    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.pixelnorm(x) if self.use_pn else x
        x = self.lrelu(self.conv2(x))
        x = self.pixelnorm(x) if self.use_pn else x
        return x
        
        
        
        
    
    
    
    
    
        
        
        