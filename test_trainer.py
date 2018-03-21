from Trainer import Trainer
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

train_x = np.loadtxt("./data/head_x.csv", delimiter=',')
train_y = np.loadtxt("./data/head_y.csv", delimiter=',')

hidden_layer1 = 5000
hidden_layer2 = 2500




model = nn.Sequential(
    nn.Linear(64*64, 2000),
    nn.ReLU(),
    nn.Linear(2000, 500),
    nn.Sigmoid(),
    nn.Linear(500, 50),
    nn.Softmax(),
    nn.Linear(50, 10)
)

# model = Net(hidden_layer1, hidden_layer2)
trainer = Trainer(model, learning_rate=1e-4, CUDA=False)
trainer.train(train_x, train_y, num_epochs=300, log=False, num_epochs_to_log=100)
print(trainer.test(train_x, train_y))

