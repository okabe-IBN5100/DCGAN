import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHANNELS = 64
EPOCHS = 1
LEARNING_RATE = 0.0002
BATCH_SIZE = 128
BETA1 = 0.5

# UPDATED HYPERPARAMETERS