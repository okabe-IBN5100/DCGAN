from model import Generator
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.utils as vutils
from config import *

fixed_noise = torch.randn(64, 100, 1, 1, device=DEVICE)

G = torch.load("Generator.pt")
fake = G(fixed_noise)
img = vutils.make_grid(fake, padding=2, normalize=True)

plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
plt.show()