import torch
import torch.nn as nn
import torch.nn.Functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import pandas as pd


torch.cuda.manual_seed(1)


train_x = pd.read_csv('./data/train_x.csv', header=None, sep=',')
loader_x = torch.utils.data.DataLoader(Variable(torch.cuda.FloatTensor(train_x)), batch_size=250, shuffle=True, num_workers=1)


train_y = pd.read_csv('./data/train_y.csv', header=None, sep=',')
loader_y = torch.utils.data.DataLoader(Variable(torch.cuda.FloatTensor(train_y)), batch_size=250, shuffle=True, num_workers=1)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2D()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)






def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))




model = Net()
model.cuda()

step_size = 1e-3
optimizer = optim.SGD(model.parameters(), lr=ste_size, momentum=0.9)


for epoch in range(100):
    