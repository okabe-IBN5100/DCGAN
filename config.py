import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHANNELS = 64
EPOCHS = 5
LEARNING_RATE = 0.002
BATCH_SIZE = 128