import torch.nn as nn


# 定义一个autoencoder模型
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 3),
            nn.Tanh(),
            nn.Linear(3, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 3),
            nn.Tanh(),
            nn.Linear(3, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder