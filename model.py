import torch
import torch.nn as nn
import numpy as np


class Conv_Block(nn.Module):
    # input_size and output_size refer to the number of color channels
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        x = self.conv_block(x)
        return x

class UNET_Model(nn.Module):
    # output_size = 59 clothing categories 
    # encoder path defined by the paper specifies feature counts in the order of 64 -> 128 -> 256 -> 512
    # decoder path feature counts is the reverse
    def __init__(self, input_size=3, output_size=59, encoder_feature_counts=[64, 128, 256, 512], decoder_feature_counts=[512, 256, 128, 64]):
        super().__init__()
        self.encoder_path, self.decoder_path, self.bottom, self.last_conv, self.pool = nn.ModuleList(), nn.ModuleList(), Conv_Block(512, 512*2), nn.Conv2d(64, 59, 1), nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in encoder_feature_counts:
            self.encoder_path.append(Conv_Block(input_size, feature))
            input_size = feature

        for feature in decoder_feature_counts:
            self.decoder_path.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)) # upsampling by a scale of 2
            self.decoder_path.append(Conv_Block(feature *2, feature))
    
    def forward(self, x):
        horizontal_connections = []
        for path_unit in self.encoder_path:
            x = path_unit(x)
            horizontal_connections.append(x)
            x = self.pool(x)

        x = self.bottom(x)

        horizontal_connections.reverse()
        for i in range(0, len(self.decoder_path), 2):
            x = self.decoder_path[i](x) # upsample the image, which will then be concatenated to the horizontal connection from the left side of the 'U'
            connection = horizontal_connections[i//2]
            concatenated = torch.cat((connection, x), dim=1)
            x = self.decoder_path[i+1](concatenated) # run the concatenated result through the conv block
        
        x = self.last_conv(x)
        return x
