import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=4, embed_dim=50, channels=3, features=64):
        super().__init__()
        
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        
        input_dim = z_dim + embed_dim
        
        self.net = nn.Sequential(
            # input: (input_dim) -> 4x4
            self._block(input_dim, features * 8, 4, 1, 0),  # 4x4
            self._block(features * 8, features * 4, 4, 2, 1),  # 8x8
            self._block(features * 4, features * 2, 4, 2, 1),  # 16x16
            self._block(features * 2, features,     4, 2, 1),  # 32x32
            nn.ConvTranspose2d(features, channels,  4, 2, 1),  # 64x64
            nn.Tanh()
        )
    
    def _block(self, in_c, out_c, kernel, stride, pad):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, z, labels):
        # z: (B, z_dim), labels: (B,)
        embed = self.label_embedding(labels)          # (B, embed_dim)
        x = torch.cat([z, embed], dim=1)              # (B, z_dim + embed_dim)
        x = x.unsqueeze(-1).unsqueeze(-1)             # (B, *, 1, 1)
        return self.net(x)

