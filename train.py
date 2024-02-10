import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from data import dataloader
from model import Generator, Discriminator, weights_init
from config import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

G_loss = []
D_loss = []
img_list = []


imgs = []

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)
G.apply(weights_init)
D.apply(weights_init)

criterion = nn.BCELoss()
optG = torch.optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

iters = 0
for epoch in range(EPOCHS):
    print(f"Training.....{epoch+1}/{EPOCHS}")
    for batchID, data in enumerate(dataloader):
        real_data = data[0].to(DEVICE)
        
        # Training Discriminator with real data

        D.zero_grad()
        batchSize = real_data.size(0)
        labels = torch.ones(batchSize, device=DEVICE) # Real data has label 1

        out = D(real_data).view(-1)
    
        real_errD = criterion(out, labels)
        real_errD.backward()

        D_x = out.mean().item()

        # Training Discriminator with fake data

        noise = torch.randn(batchSize, 100, 1, 1, device=DEVICE)
        labels.fill_(0)

        fake = G(noise)
        out = D(fake.detach()).view(-1)
        fake_errD = criterion(out, labels)
        fake_errD.backward()

        errD = real_errD + fake_errD
        optD.step()

        D_G_z1 = out.mean().item()

        # Training Generator

        G.zero_grad()
        labels.fill_(1)
        out = D(fake).view(-1)
        errG = criterion(out, labels)
        errG.backward()
        optG.step()

        D_G_z2 = out.mean().item()

        if batchID % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch+1, EPOCHS, batchID, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_loss.append(errG.item())
        D_loss.append(errD.item())

        iters += 1


plt.plot(D_loss, label="Discriminator")
plt.plot(G_loss, label="Generator")
plt.legend()
plt.show()

torch.save(G, "Generator.pt")



