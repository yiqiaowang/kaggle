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


CUDA = False


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

data_y = np.loadtxt(URL_ENDPOINT+"train_y_30000.csv", delimiter=",")

print("y loaded")
data_x = np.loadtxt(URL_ENDPOINT+"train_x_30000.csv", delimiter=",")

print("x loaded")
x_train = torch.FloatTensor(data_x)
y_train = torch.FloatTensor(convert_y(data_y))


batch = 100
num_epochs = 5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
hidden_layer1_range = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000]
hidden_layer2_range = [2000, 1800, 1600, 1400, 1200,1000,800, 600, 400, 200]
# hidden_layer1_range = [5000]
# hidden_layer2_range = [200]

accuracy_list = []
#hidden_layer1 = 5000
#hidden_layer2 = 500

class Net(nn.Module):
    def __init__(self, hidden_layer1, hidden_layer2):
        super(Net, self).__init__()
        print("called super")
        self.filter_size = 5
        self.conv_filters1 = 6
        self.conv_filters2 = 16
        self.output_layer = 10
        self.input_size = 64
        self.stride1 = 1
        self.stride2 = 1

        self.hidden_layer0 = self.conv_filters2 * (self.input_size - 2*self.filter_size + 2)**2
        self.conv1 = nn.Conv2d(1, self.conv_filters1, self.filter_size, stride = self.stride1)
        # self.pool = nn.MaxPool1d(2, 2) -> NOT USED AT THE MOMENT
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

criterion = nn.MSELoss()
train = torch.utils.data.TensorDataset(x_train, y_train)
trainloader = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=1)
for hidden_layer1 in hidden_layer1_range:
    for hidden_layer2 in hidden_layer2_range:
        print("About to train on hidden layer 1 =", hidden_layer1, " and hidden layer 2 =", hidden_layer2)
        net = Net(hidden_layer1, hidden_layer2)
        if CUDA:
            net = net.cuda()
        print("created net")
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):

                # wrap them in Variable
                if CUDA:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

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
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

        y_test = np.loadtxt(URL_ENDPOINT+"test_y_10000.csv", delimiter=",")
        x_test = np.loadtxt(URL_ENDPOINT+"test_x_10000.csv", delimiter=",")
        x_test = Variable(torch.FloatTensor(x_test))
        if CUDA:
            x_test = x_test.cuda()
        output = net(x_test)
        output = output.data
        if CUDA:
            output = output.cpu()
        output = convert_to_label(output.numpy())
        accuracy = accuracy_score(y_test, output)
        accuracy_list.append((accuracy, hidden_layer1, hidden_layer2))
        print("with hidden layer 1 of {} and hidden layer 2 of {} gives a test accuracy of {}".format(hidden_layer1, hidden_layer2, accuracy))
end = time()
for i in accuracy_list:
    print(i)
print('Finished Training. Took {} seconds to compute'.format(end-start))