import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
from torchvision.models import EfficientNet_V2_S_Weights,EfficientNet_V2_M_Weights,EfficientNet_V2_L_Weights
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






def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    else:
        return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                             conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0))


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UNetDownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = conv(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UNetUpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv(2 * self.out_channels, self.out_channels)
        else:
            self.conv1 = conv(self.out_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, from_up, from_down):
        from_up = self.upconv(from_up)

        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x


class UNet(nn.Module):
    def __init__(self, n_channels=3, merge_mode='concat', up_mode='transpose'):
        super(UNet, self).__init__()
        self.n_chnnels = n_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.down1 = UNetDownBlock(self.n_chnnels, 64, 3, 1, 1)
        self.down2 = UNetDownBlock(64, 128, 4, 2, 1)
        self.down3 = UNetDownBlock(128, 256, 4, 2, 1)
        self.down4 = UNetDownBlock(256, 512, 4, 2, 1)
        self.down5 = UNetDownBlock(512, 512, 4, 2, 1)

        self.up1 = UNetUpBlock(512, 512, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up2 = UNetUpBlock(512, 256, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up3 = UNetUpBlock(256, 128, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up4 = UNetUpBlock(128, 64, merge_mode=self.merge_mode, up_mode=self.up_mode)

        self.conv_final = nn.Sequential(conv(64, 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv_final(x)

        return x