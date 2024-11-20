import timm
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
from torchvision.models import EfficientNet_V2_S_Weights,EfficientNet_V2_M_Weights,EfficientNet_V2_L_Weights
from torchvision.models import swin_b, Swin_B_Weights
from .swin_autoencoder import SwinTAE



class Decoder(nn.Module):
    def __init__(self, channels, blocks, kernel_size=4, stride=2, padding=1):
        super(Decoder, self).__init__()
        
        def up_block(in_channels, out_channels, act=nn.ReLU()):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                act
            )
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )
            
        layers = []
        for i in range(len(channels) - 1):
            for j in range(blocks[i]):
                if j == blocks[i] - 1:
                    if i == len(channels) - 2:
                        layers.append(up_block(channels[i], channels[i+1], nn.Sigmoid()))
                    else:
                        layers.append(up_block(channels[i], channels[i+1]))
                else:
                    layers.append(conv_block(channels[i],channels[i]))
                
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.decoder(x)


class EfficientNetV2SAutoencoder(nn.Module):
    def __init__(self):
        super(EfficientNetV2SAutoencoder, self).__init__()
        
        self.encoder = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT)
        self.decoder = Decoder(channels=[1280, 640, 256, 128, 64, 3],
                               blocks=[1,1,1,1,1])
        
    def forward(self, x):
        x = self.encoder.features(x)
        x = self.decoder(x)
        return x
    
    
class EfficientNetUNet(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNetUNet, self).__init__()
        
        # EfficientNet Encoder
        encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.encoder_layers = nn.ModuleList(encoder.features)  # EfficientNet features layers
        
        # Output channel sizes from each encoder layer based on your provided sizes
        self.selected_layers = [0, 2, 3, 5, 8]  # 각 레이어의 출력이 중요하다고 판단될 경우 선택
        encoder_out_channels = [32, 24, 40, 112, 1280]
        # encoder_out_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]

        # U-Net style decoder (UpConvs and Skip Connections)
        self.up_convs = nn.ModuleList([
            nn.ConvTranspose2d(encoder_out_channels[i], encoder_out_channels[i - 1], kernel_size=2, stride=2)
            for i in range(len(encoder_out_channels) - 1, 0, -1)
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_out_channels[i-1] + encoder_out_channels[i-1], encoder_out_channels[i-1], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(encoder_out_channels[i-1], encoder_out_channels[i-1], kernel_size=3, padding=1),
                nn.ReLU()
            ) for i in range(len(encoder_out_channels) - 1, 0, -1)
        ])

        # Final Convolution
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(encoder_out_channels[0], 3, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(3, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder
        enc_outputs = []
        for idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if idx in self.selected_layers:  # 선택한 단계만 저장
                enc_outputs.append(x)

        # Decoder with skip connections
        for i in range(len(self.decoders)):
            x = self.up_convs[i](x)
            x = torch.cat([x, enc_outputs[-(i + 2)]], dim=1)  # Skip connection
            x = self.decoders[i](x)

        # Final Convolution
        x = self.final_conv(x)
        return x

class EfficientNetB0Unet(nn.Module):
    def __init__(self):
        super(EfficientNetB0Unet, self).__init__()
        self.unet_model = smp.Unet(encoder_name='efficientnet-b0')
        self.unet_model.segmentation_head[0] = nn.Conv2d(16,3,3,1,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.unet_model(x)
        x = self.sigmoid(x)
        return x


    