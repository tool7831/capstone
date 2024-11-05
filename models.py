import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
from torchvision.models import EfficientNet_V2_S_Weights,EfficientNet_V2_M_Weights,EfficientNet_V2_L_Weights
import timm

# EfficientNet 기반 Autoencoder 정의
class EfficientNetB0Autoencoder(nn.Module):
    def __init__(self):
        super(EfficientNetB0Autoencoder, self).__init__()
        # EfficientNet-b0을 encoder로 사용
        self.encoder = efficientnet_b0(EfficientNet_B0_Weights.DEFAULT)
        
        # Decoder 정의
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # pixel 값을 [0, 1] 범위로 맞추기 위해 사용
        )

    def forward(self, x):
        # Encoder를 통해 특징 추출
        x = self.encoder.features(x)
        # Decoder를 통해 재구성
        x = self.decoder(x)
        return x
    

# EfficientNet 기반 Autoencoder 정의
class EfficientNetB1Autoencoder(nn.Module):
    def __init__(self):
        super(EfficientNetB1Autoencoder, self).__init__()
        # EfficientNet-b1을 encoder로 사용
        self.encoder = efficientnet_b1(EfficientNet_B1_Weights.DEFAULT)
        
        # Decoder 정의
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 640, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(640, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # pixel 값을 [0, 1] 범위로 맞추기 위해 사용
        )

    def forward(self, x):
        # Encoder를 통해 특징 추출
        x = self.encoder.features(x)
        # Decoder를 통해 재구성
        x = self.decoder(x)
        return x
    
# EfficientNet 기반 Autoencoder 정의
class EfficientNetB2Autoencoder(nn.Module):
    def __init__(self):
        super(EfficientNetB2Autoencoder, self).__init__()
        self.encoder = efficientnet_b2(EfficientNet_B2_Weights.DEFAULT)
        
        # Decoder 정의
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1408, 640, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(640, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # pixel 값을 [0, 1] 범위로 맞추기 위해 사용
        )

    def forward(self, x):
        # Encoder를 통해 특징 추출
        x = self.encoder.features(x)
        # Decoder를 통해 재구성
        x = self.decoder(x)
        return x

# EfficientNet 기반 Autoencoder 정의
class EfficientNetB3Autoencoder(nn.Module):
    def __init__(self):
        super(EfficientNetB3Autoencoder, self).__init__()
        # EfficientNet-b1을 encoder로 사용
        self.encoder = efficientnet_b3(EfficientNet_B3_Weights.DEFAULT)
        
        # Decoder 정의
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1536, 768, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(384, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # pixel 값을 [0, 1] 범위로 맞추기 위해 사용
        )

    def forward(self, x):
        # Encoder를 통해 특징 추출
        x = self.encoder.features(x)
        # Decoder를 통해 재구성
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
    
class EfficientNetB1Unet(nn.Module):
    def __init__(self):
        super(EfficientNetB1Unet, self).__init__()
        self.unet_model = smp.Unet(encoder_name='efficientnet-b1')
        self.unet_model.segmentation_head[0] = nn.Conv2d(16,3,3,1,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.unet_model(x)
        x = self.sigmoid(x)
        return x
    

class EfficientNetB2Unet(nn.Module):
    def __init__(self):
        super(EfficientNetB2Unet, self).__init__()
        self.unet_model = smp.Unet(encoder_name='efficientnet-b2')
        self.unet_model.segmentation_head[0] = nn.Conv2d(16,3,3,1,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.unet_model(x)
        x = self.sigmoid(x)
        return x
    

class EfficientNetB3Unet(nn.Module):
    def __init__(self):
        super(EfficientNetB3Unet, self).__init__()
        self.unet_model = smp.Unet(encoder_name='efficientnet-b3')
        self.unet_model.segmentation_head[0] = nn.Conv2d(16,3,3,1,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.unet_model(x)
        x = self.sigmoid(x)
        return x

class EfficientNetV2SAutoencoder(nn.Module):
    def __init__(self):
        super(EfficientNetV2SAutoencoder, self).__init__()
        self.encoder = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT)
        
        # Decoder 정의
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 640, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(640, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # pixel 값을 [0, 1] 범위로 맞추기 위해 사용
        )

    def forward(self, x):
        # Encoder를 통해 특징 추출
        x = self.encoder.features(x)
        # Decoder를 통해 재구성
        x = self.decoder(x)
        return x
    
class EfficientNetV2MAutoencoder(nn.Module):
    def __init__(self):
        super(EfficientNetV2MAutoencoder, self).__init__()
        self.encoder = efficientnet_v2_m(EfficientNet_V2_M_Weights.DEFAULT)
        
        # Decoder 정의
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 640, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(640, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # pixel 값을 [0, 1] 범위로 맞추기 위해 사용
        )

    def forward(self, x):
        # Encoder를 통해 특징 추출
        x = self.encoder.features(x)
        # Decoder를 통해 재구성
        x = self.decoder(x)
        return x
    
    
class EfficientNetV2LAutoencoder(nn.Module):
    def __init__(self):
        super(EfficientNetV2LAutoencoder, self).__init__()
        self.encoder = efficientnet_v2_l(EfficientNet_V2_L_Weights.DEFAULT)
        
        # Decoder 정의
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 640, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(640, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # pixel 값을 [0, 1] 범위로 맞추기 위해 사용
        )

    def forward(self, x):
        # Encoder를 통해 특징 추출
        x = self.encoder.features(x)
        # Decoder를 통해 재구성
        x = self.decoder(x)
        return x
    
    
class EfficientNetB0AutoencoderC(nn.Module):
    def __init__(self):
        super(EfficientNetB0AutoencoderC, self).__init__()
        self.encoder = efficientnet_b0(EfficientNet_B0_Weights.DEFAULT)
        
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        self.decoder = nn.Sequential(
            CBR(1280, 512),
            CBR(512, 256),
            CBR(256, 128),
            CBR(128, 64),
            CBR(64, 3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder를 통해 특징 추출
        x = self.encoder.features(x)
        # Decoder를 통해 재구성
        x = self.decoder(x)
        return x

class EfficientNetB2AutoencoderC(nn.Module):
    def __init__(self):
        super(EfficientNetB2AutoencoderC, self).__init__()
        self.encoder = efficientnet_b2(EfficientNet_B2_Weights.DEFAULT)
        
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        self.decoder = nn.Sequential(
            CBR(1408, 704),
            CBR(704, 302),
            CBR(302, 128),
            CBR(128, 64),
            CBR(64, 3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder를 통해 특징 추출
        x = self.encoder.features(x)
        # Decoder를 통해 재구성
        x = self.decoder(x)
        return x

