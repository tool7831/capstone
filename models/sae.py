import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
from models.autoencoder import Decoder


def create_noise(true_feats, mix_noise=1, noise_std=0.1, device='cuda'):
    B, C, H, W = true_feats.shape
    # 1. 노이즈 인덱스 생성 (B 크기)
    noise_idxs = torch.randint(0, mix_noise, size=(B,))
    # 2. 원-핫 인코딩 (B, K)
    noise_one_hot = F.one_hot(noise_idxs, num_classes=mix_noise)

    # 3. 가우시안 노이즈 생성 (B, K, C, H, W)
    noise = torch.stack([
        torch.normal(0, noise_std * 1.1**k, size=(B, C, H, W))
        for k in range(mix_noise)
    ], dim=1)

    noise = (noise * noise_one_hot.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

    return noise.to(device)

def delete_feature(true_feats, delete_rate=0.01, device='cuda'):
    B, C, H, W = true_feats.shape
    # channel_idx = torch.tensor(torch.bernoulli((1-delete_rate) * torch.ones(C)), dtype=torch.float32).to(device)  # 1-delete_rate 확률로 1, delete_rate 확률로 0
    # fake_feats = true_feats * channel_idx.view(1, C, 1, 1)
    
    spatial_idx = torch.tensor(torch.bernoulli((1-delete_rate) * torch.ones(H*W)), dtype=torch.float32).to(device)
    fake_feats = true_feats * spatial_idx.view(1, 1, H, W)
    
    return fake_feats


class SimpleSAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT).features
        self.feature_adaptor = nn.Linear(1280, 1280)
        self.decoder = Decoder(channels=[1280, 640, 256, 128, 64, 4],
                               blocks=[2,2,2,2,1])
    
    def forward(self, x):
        
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = x.permute(0, 2, 3, 1)
        x = self.feature_adaptor(x)
        x = x.permute(0, 3, 1, 2)
        x = self.decoder(x)
        mask = x[:,3]
        x = x[:,0:3]
        return x, mask
    
    def train_model(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = x.permute(0, 2, 3, 1)   # B, H, W, C
        x = self.feature_adaptor(x) 
        x = x.permute(0, 3, 1, 2)   # B, C, H, W
        noise = create_noise(x, device=self.device)
        x = x + noise
        x = self.decoder(x)
        mask = x[:,3]
        x = x[:,0:3]
        return x, mask
    
    
class SimpleDAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT).features
        # self.feature_adaptor = nn.Linear(1280, 1280)
        self.decoder = Decoder(channels=[1280, 640, 256, 128, 64, 3],
                               blocks=[1,1,1,1,1])
        self.noise_prediction = nn.Sequential(
            nn.Conv2d(1280, 1280, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(1280, 1280, 3, 1, 1),
        )
    
    def forward(self, x):
        # with torch.no_grad():
        x = self.feature_extractor(x)
        # x = x.permute(0, 2, 3, 1)
        # x = self.feature_adaptor(x)
        # x = x.permute(0, 3, 1, 2)
        _noise = self.noise_prediction(x)
        x = x - _noise
        x = self.decoder(x)
        return x
    
    def train_model(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = x.permute(0, 2, 3, 1)   # B, H, W, C
        x = self.feature_adaptor(x) 
        x = x.permute(0, 3, 1, 2)   # B, C, H, W
        if np.random.rand() < 0.5:
            noise = create_noise(x)
            noise = delete_feature(noise, 0.99)
        else:
            noise = torch.zeros_like(x, device=self.device, requires_grad=False)
        
        x = x + noise
        _noise = self.noise_prediction(x)
        loss = F.l1_loss(noise, _noise)
        loss.backward()
        
        x = self.decoder(x)     

        
        return x