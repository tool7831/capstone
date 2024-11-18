import torch
import torch.nn as nn
from torchvision.models import swin_b, Swin_B_Weights
import timm

class PatchExpansion(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim//2)
        self.expand = nn.Linear(dim, 2*dim, bias=False)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H , W, 2, 2, C//4)
        x = x.permute(0,1,3,2,4,5)
        
        x = x.reshape(B,H*2, W*2 , C//4)
        
        x = self.norm(x)
        return x


class SwinDecoder(nn.Module):
    def __init__(self):
        super(SwinDecoder, self).__init__()
        self.expansion = nn.Sequential(
            PatchExpansion(1024),
            nn.ReLU(),
            PatchExpansion(512),
            nn.ReLU(),
            PatchExpansion(256),
            nn.ReLU(),
            PatchExpansion(128),
            nn.ReLU(),
            PatchExpansion(64),
            nn.ReLU(),
        ) 
        self.head = nn.Linear(32, 3)
        
    def forward(self, x):
        x = self.expansion(x) # B, 224, 224, 32
        x = self.head(x)
        x = x.permute(0,3,1,2)
        return x
        

class SwinTAE(nn.Module):
    def __init__(self):
        super(SwinTAE, self).__init__()
        self.encoder = swin_b(Swin_B_Weights.DEFAULT).features
        self.decoder = SwinDecoder()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

