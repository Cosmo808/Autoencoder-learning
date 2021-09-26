import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        # N, 784
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128), nn.ReLU(),  # N, 784 -> N, 128
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64, 12), nn.ReLU(),
            nn.Linear(12, 3)  # -> N, 3
        )
        self.decoder=nn.Sequential(
            nn.Linear(3,12),nn.ReLU(),
            nn.Linear(12,64),nn.ReLU(),
            nn.Linear(64,128),nn.ReLU(),
            nn.Linear(128,28*28),
            nn.Sigmoid()  # -> N, 3 -> N, 784
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
