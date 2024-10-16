from torch import nn
import segmentation_models_pytorch as smp

class AEUnet(nn.Module):
    def __init__(self, use_pretrained=True, resnet_version="resnet18"):
        super(AEUnet, self).__init__()

        weights = "imagenet" if use_pretrained else None

        self.resnet = smp.Unet(
            encoder_name=resnet_version,
            encoder_weights=weights,
            classes=1
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # DeepConvBlock(256, 3, 256),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # DeepConvBlock(128, 3, 128),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # DeepConvBlock(64, 3, 64),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # DeepConvBlock(64, 3, 64),

            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # DeepConvBlock(1, 3, 1),
        )

    def forward(self, x):
        return self.segment(x)

    def segment(self, x):
        x = self.resnet(x)
        return x

    def reconstruct(self, x):
        x = self.resnet.encoder(x)
        x = x[-1]
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.resnet.encoder(x)
        return x

    def both(self, x):
        x = self.resnet.encoder(x)
        x = x[-1]
        seg = self.resnet.decoder(x)
        rec = self.decoder(x)
        return seg, rec

class DeepConvBlock(nn.Module):
    # This is a block to quickly build deeper networks
    def __init__(self, in_channels, number_layers, out_channels):
        super(DeepConvBlock, self).__init__()
        layers = []
        for i in range(number_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.ReLU())

        out_conv = []
        out_conv.append(nn.BatchNorm2d(in_channels))
        out_conv.append(nn.ReLU())
        out_conv.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        out_conv.append(nn.BatchNorm2d(out_channels))
        out_conv.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.out_conv = nn.Sequential(*out_conv)

    def forward(self, x):
        residual = x  # Save the input for the residual connection
        block = self.layers(x)
        block = block + residual  # Add the residual connection
        return self.out_conv(block)