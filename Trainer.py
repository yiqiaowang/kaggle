import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader


class Trainer:
    def __init__(self, model, learning_rate=1e-4, momentum=0.9, CUDA=False):
        self.model = model
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.cuda = CUDA and torch.cuda.is_available()
        self.model = model
        if self.cuda:
            self.model = self.model.cuda()


    def train(self, train_x, train_y, num_epochs, log=True, batch_size=100, num_steps_to_log=100):
        training_data = torch.FloatTensor(train_x.reshape(-1, 64, 64))
        training_y = Trainer.convert_to_y(train_y)
        dataset = TensorDataset(training_data, training_y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        for epoch in range(num_epochs):
            for batch, (inputs, labels) in enumerate(dataloader):
                if self.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if log and batch % (num_steps_to_log - 1) == 0:
                    print("epoch {}, batch {}, loss {}".format(epoch+1, batch+1, loss.data[0]))
            
            if log:
                print("Finished epoch {} with loss of {}".format(epoch+1, loss.data[0]))
        return self.model


    def test(self, test_x, test_y):
        test_x = torch.FloatTensor(test_x.reshape(-1, 64, 64))
        if self.cuda:
            test_x = test_x.cuda()

        output = self.model(test_x)
        output = output.data
        if self.cuda:
            output = output.cpu()
        accuracy = accuracy_score(y_test, output)
        return accuracy



    def convert_to_y(labels):
        output = []
        for label in labels:
            y_vector = np.zeros(10)
            y_vector[int(label)] = 1
            output.append(y_vector)
        return torch.FloatTensor(output)



