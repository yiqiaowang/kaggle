import torch
import numpy as np
import scipy.misc
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import accuracy_score
from time import time


def convert_y(y):
    output = []
    for i in y:
        to_add = np.zeros(10, dtype='int')
        to_add[int(i)] = 1
        output.append(to_add)
    return np.array(output)


def convert_to_label(data):
    output = []
    for d in data:
        output.append(np.argmax(d))
    return np.array(output).astype(int)
start = time()
URL_ENDPOINT = "http://cs.mcgill.ca/~pbelan17/"

data_y = np.loadtxt(URL_ENDPOINT+"train_y_20000.csv", delimiter=",")
y_test = np.loadtxt(URL_ENDPOINT+"test_y_500.csv", delimiter=",")
print("y loaded")
data_x = np.loadtxt(URL_ENDPOINT+"train_x_20000.csv", delimiter=",")
x_test = np.loadtxt(URL_ENDPOINT+"test_x_500.csv", delimiter=",")
print(x_test.shape)
print("x loaded")
print(data_y.shape)
print(data_x.shape)
x_train = torch.FloatTensor(data_x)
y_train = torch.FloatTensor(convert_y(data_y))
x_test = Variable(torch.FloatTensor(x_test).cuda())

batch = 10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        filter_size = 5
        conv_filters1 = 6
        conv_filters2 = 16
        hidden_layer1 = 5000
        hidden_layer2 = 500
        output_layer = 10
        input_size = 64
        stride1 = 1
        stride2 = 1

        hidden_layer0 = conv_filters2 * (input_size - 2*filter_size + 2)^2

        self.conv1 = nn.Conv2d(1, conv_filters1, filter_size, stride = stride1)
        # self.pool = nn.MaxPool1d(2, 2) -> NOT USED AT THE MOMENT
        self.conv2 = nn.Conv2d(conv_filters1, conv_filters2, filter_size, stride = stride2)
        self.fc1 = nn.Linear(hidden_layer0 , hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc3 = nn.Linear(hidden_layer2, output_layer)

    def forward(self, x):
        x = self.conv1(x.view(-1,1,input_size,input_size))
        x = F.relu(x)
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.hidden_layer0)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
train = torch.utils.data.TensorDataset(x_train, y_train)
trainloader = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=1)

for epoch in range(3):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]

        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            print(inputs.shape)
            running_loss = 0.0

print(x_test.shape)
output = net(x_test)
output = output.data.cpu()
output = convert_to_label(output.numpy())
accuracy = accuracy_score(y_test, output)
print(accuracy)
end = time()
print('Finished Training. Took {} seconds to compute'.format(end-start))