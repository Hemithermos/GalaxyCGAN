import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, num_classes=4, channels=3, features=64, image_size=64):
        super().__init__()
        
        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)
        self.image_size = image_size
        
        self.net = nn.Sequential(
            self._block(channels + 1, features,     4, 2, 1, bn=False),  # 32x32
            self._block(features,     features * 2, 4, 2, 1),            # 16x16
            self._block(features * 2, features * 4, 4, 2, 1),            # 8x8
            self._block(features * 4, features * 8, 4, 2, 1),            # 4x4
            spectral_norm(nn.Conv2d(features * 8, 1, 4, 1, 0)),          # 1x1
        )
    
    def _block(self, in_c, out_c, kernel, stride, pad, bn=True):
        layers = [spectral_norm(nn.Conv2d(in_c, out_c, kernel, stride, pad, bias=False))]
        if bn:
            layers.append(nn.InstanceNorm2d(out_c, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x, labels):
        embed = self.label_embedding(labels)
        embed = embed.view(-1, 1, self.image_size, self.image_size)
        x = torch.cat([x, embed], dim=1)
        return self.net(x).squeeze()

