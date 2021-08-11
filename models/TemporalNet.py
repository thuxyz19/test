import numpy as np
import torch
import torch.nn as nn
from mvn_new.models.v2v import Basic3DBlock, Res3DBlock, Pool3DBlock, Upsample3DBlock


BN_MOMENTUM = 0.1


class GlobalAveragePoolingHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=1, padding=1),
            nn.BatchNorm3d(32, momentum=BN_MOMENTUM),
            nn.MaxPool3d((1, 2, 2)),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm3d(32, momentum=BN_MOMENTUM),
            nn.MaxPool3d((1, 2, 2)),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        frames = x.shape[2]
        x = self.features(x)

        batch_size, n_channels = x.shape[:2]
        x = x.view((batch_size, n_channels, frames, -1))
        x = x.mean(dim=-1)
        x = x.transpose(1, 2).contiguous().view(-1, n_channels)
        out = self.head(x)

        return out


class EncoderDecorder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128)
        self.encoder_pool3 = Pool3DBlock(2)
        self.encoder_res3 = Res3DBlock(128, 128)

        self.mid_res = Res3DBlock(128, 128)

        self.decoder_res3 = Res3DBlock(128, 128)
        self.decoder_upsample3 = Upsample3DBlock(128, 128, 2, 2)
        self.decoder_res2 = Res3DBlock(128, 128)
        self.decoder_upsample2 = Upsample3DBlock(128, 64, 2, 2)
        self.decoder_res1 = Res3DBlock(64, 64)
        self.decoder_upsample1 = Upsample3DBlock(64, 32, 2, 2)

        self.skip_res1 = Res3DBlock(32, 32)
        self.skip_res2 = Res3DBlock(64, 64)
        self.skip_res3 = Res3DBlock(128, 128)

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        skip_x3 = self.skip_res3(x)
        x = self.encoder_pool3(x)
        x = self.encoder_res3(x)

        x = self.mid_res(x)

        x = self.decoder_res3(x)
        x = self.decoder_upsample3(x)
        x = x + skip_x3
        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1

        return x


class TemporalNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 16, 7),
            Res3DBlock(16, 32),
            Res3DBlock(32, 32))

        self.encoder_decoder = EncoderDecorder()

        self.back_layers = nn.Sequential(
            Res3DBlock(32, 32),
            Basic3DBlock(32, 32, 1))

        self.alg_confidence = GlobalAveragePoolingHead(32, 17)

        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.back_layers(x)
        confidence = self.alg_confidence(x)
        x = self.output_layer(x)
        return x, confidence

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)

