import torch
import torch.nn as nn
from config import CHANNELS

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, CHANNELS*16, 4, 1, 0), # (1, 100) -> (64*16, 4, 4)
            nn.ReLU(),

            nn.ConvTranspose2d(CHANNELS*16, CHANNELS*8, 4, 2, 1), # (64*16, 4, 4) -> (64*8, 8, 8)
            nn.ReLU(),

            nn.ConvTranspose2d(CHANNELS*8, CHANNELS*4, 4, 2, 1), # (64*8, 8, 8) -> (64*4, 16,16)
            nn.ReLU(),

            nn.ConvTranspose2d(CHANNELS*4, CHANNELS*2, 4, 2, 1), # (64*4, 16, 16) -> (64*2, 32, 32)
            nn.ReLU(),

            nn.ConvTranspose2d(CHANNELS*2, 3, 4, 2, 1), # (64*2, 32, 32) -> (3, 64, 64)
            nn.ReLU()
        )

    def forward(self, X):
        return self.model(X)
    
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, CHANNELS, 4, 2, 1), # (3, 64, 64) -> (64, 32, 32)
            nn.ReLU(),

            nn.Conv2d(CHANNELS, CHANNELS*2, 4, 2, 1), # (64, 32, 32) -> (128, 16, 16)
            nn.ReLU(),

            nn.Conv2d(CHANNELS*2, CHANNELS*4, 4, 2, 1), # (128, 16, 16) -> (256, 8, 8)
            nn.ReLU(),

            nn.Conv2d(CHANNELS*4, CHANNELS*8, 4, 2, 1), # (256, 8, 8) -> (512, 4, 4)
            nn.ReLU(),

            nn.Conv2d(CHANNELS*8, 1, 4, 1, 0), # (512, 4, 4) -> (1, 1, 1)
            nn.Sigmoid(),
        )

    def forward(self, X):
        return self.model(X)
    

if __name__ == "__main__":
    # TESTING GENERATOR AND DISCRIMINATOR
    
    X = torch.rand(100, 1, 1)

    G = Generator()
    D = Discriminator()

    print(D(G(X)).shape)