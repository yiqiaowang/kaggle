from Trainer import Trainer
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

train_x = np.loadtxt("./data/head_x.csv", delimiter=',')
train_y = np.loadtxt("./data/head_y.csv", delimiter=',')

hidden_layer1 = 5000
hidden_layer2 = 2500

class Net(nn.Module):
    def __init__(self, hidden_layer1, hidden_layer2):
        super(Net, self).__init__()
        self.filter_size = 5
        self.conv_filters1 = 6
        self.conv_filters2 = 16
        self.output_layer = 10
        self.input_size = 64
        self.stride1 = 1
        self.stride2 = 1

        self.hidden_layer0 = self.conv_filters2 * (self.input_size - 2*self.filter_size + 2)**2
        self.conv1 = nn.Conv2d(1, self.conv_filters1, self.filter_size, stride = self.stride1)
        self.conv2 = nn.Conv2d(self.conv_filters1, self.conv_filters2, self.filter_size, stride = self.stride2)
        self.fc1 = nn.Linear(self.hidden_layer0 , hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc3 = nn.Linear(hidden_layer2, self.output_layer)

    def forward(self, x):
        x = self.conv1(x.view(-1,1,self.input_size,self.input_size))
        x = F.relu(x)
        x = F.relu(self.conv2(x))
        x = x.view(-1,self.hidden_layer0)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net(hidden_layer1, hidden_layer2)
trainer = Trainer(model, CUDA=False)
trainer.train(train_x, train_y, num_epochs=1, num_steps_to_log=10)
print(trainer.test(train_x, train_y))

