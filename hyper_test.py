from Trainer import Trainer
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

model = nn.Sequential(
    nn.Linear(64*64, 2000),
    nn.ReLU(),
    nn.Linear(2000, 10)
)


