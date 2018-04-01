import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from preprocessing import IMAGE_SIZE

class Trainer:
    def __init__(self, model, learning_rate=1e-4, momentum=0.9, CUDA=True):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.cuda = CUDA and torch.cuda.is_available()
        self.model = model
        if self.cuda:
            self.model = self.model.cuda()


    def train(self, train_x, train_y, num_epochs, log=True, batch_size=100, num_steps_to_log=100):
        training_data = torch.FloatTensor(train_x.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE).astype(np.float))
        training_y = Trainer.convert_to_y(train_y.astype(np.uint8))
        dataset = TensorDataset(training_data, training_y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(num_epochs):
            for batch, (inputs, labels) in enumerate(dataloader):
                if self.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)
            

                optimizer.zero_grad()
                outputs = self.model(inputs)
                               
                _, classes = torch.max(labels, 1)

                
                loss = self.criterion(outputs, classes)
                loss.backward()
                optimizer.step()

                if log and batch % (num_steps_to_log - 1) == 0:
                    print("epoch {}, batch {}, loss {}".format(epoch+1, batch+1, loss.data[0]))
        
            
            if log:
                print("Finished epoch {} with loss of {}".format(epoch+1, loss.data[0]))
        return self.model


    def test(self, test_x, test_y):
        test_x = torch.FloatTensor(test_x.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE).astype(np.uint8))
        if self.cuda:
            test_x = test_x.cuda()
        test_x = Variable(test_x)
        output = self.model(test_x)
        output = output.data
        if self.cuda:
            output = output.cpu()
        output = output.numpy()
        labels = Trainer.convert_to_labels(output)
        test_y = test_y.astype(np.uint8)
        accuracy = accuracy_score(test_y, labels)
        precision = precision_score(test_y, labels, average='micro')
        recall = recall_score(test_y, labels, average='micro')
        f1 = f1_score(test_y, labels, average='micro')
        conf_matrix = confusion_matrix(test_y, labels)
        return accuracy, precision, recall, f1, conf_matrix


    def convert_to_y(labels):
        output = []
        for label in labels:
            y_vector = [0] * 10
            y_vector[int(label)] = 1
            output.append(y_vector)
        return torch.FloatTensor(output)


    def convert_to_labels(ys):
        output = []
        for y in ys:
            output.append(np.argmax(y))
        return np.array(output, dtype='int')