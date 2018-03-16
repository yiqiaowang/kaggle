import torch
from torch.autograd import Variable
import torch.utils.data
import numpy as np
from sklearn.metrics import accuracy_score


# BATCH_SIZE = 250
INPUT_DIMENSION = 64*64
HIDDEN_1 = 1000
HIDDEN_2 = 200
OUT_DIMENSION = 10

CUDA = True







def convert_y(y):
    output = []
    for i in y:
        to_add = np.zeros(10, dtype='int')
        to_add[int(i)] = 1
        output.append(to_add)
    return torch.FloatTensor(np.array(output))

def convert_to_label(data):
    output = []
    for d in data:
        output.append(np.argmax(d))
    return np.array(output).astype(int)


x = np.loadtxt("./data/train_x.csv", delimiter=',')
x = torch.FloatTensor(x)
# if CUDA:
#     x = Variable(torch.cuda.FloatTensor(x))
# else:
#     x = Variable(torch.FloatTensor(x))

y = np.loadtxt("./data/train_y.csv", delimiter=',')
y = torch.FloatTensor(y)
# y_vectors = convert_y(y)
# y_vectors = torch.FloatTensor(y_vectors)
# if CUDA:
#     y_vectors = Variable(torch.cuda.FloatTensor(y_vectors), requires_grad=False)
# else:
#     y_vectors = Variable(torch.FloatTensor(y_vectors), requires_grad=False)


data = torch.utils.data.TensorDataset(x, y)
data_loader = torch.utils.data.DataLoader(data, batch_size=250, shuffle=True, num_workers=1)

# loader_x = torch.utils.data.DataLoader(x, batch_size=250, shuffle=True, num_workers=1)
# loader_y = torch.utils.data.DataLoader(y, batch_size=250, shuffle=True, num_workers=1)

print("Data loaded!")

model = torch.nn.Sequential(
    torch.nn.Linear(INPUT_DIMENSION, HIDDEN_1),
    torch.nn.Sigmoid(),
    torch.nn.Linear(HIDDEN_1, HIDDEN_2),
    torch.nn.Sigmoid(),
    torch.nn.Linear(HIDDEN_2, OUT_DIMENSION)
)

print("created model")

if CUDA:
    model = model.cuda()

print("moved to gpu")

loss_function = torch.nn.MSELoss(size_average=False)

print("created loss function")

step_size = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=step_size)
print("created optimizer")
num_steps = 1000

from time import time

start = time()





for step in range(num_steps):
    model.train()
    for batch_id, (data, labels) in enumerate(data_loader):
        labels = convert_y(labels)
        if CUDA:
            data, labels = data.cuda(), labels.cuda()
        data, labels = Variable(data), Variable(labels)

        optimizer.zero_grad()

        predicted_y = model(data)
        loss = loss_function(predicted_y, labels)

        # if batch_id % 10 == 0:
        #     print("step {}, batch {}, loss {}".format(step, batch_id, loss.data[0]))


        loss.backward()
        optimizer.step()

    print(step, loss.data[0])


end = time()

print("It took ", (end-start), "seconds to train")

# train_output = convert_to_label(predicted_y.cpu().data.numpy())
# train_accuracy = accuracy_score(y, train_output)
# print("train accuracy is", train_accuracy)


test_x = np.loadtxt("./data/tail_x.csv", delimiter=',')
test_x = Variable(torch.cuda.FloatTensor(test_x))
test_y = np.loadtxt("./data/tail_y.csv", delimiter=',')

test_predicted_y = model(test_x)
output = convert_to_label(test_predicted_y.cpu().data.numpy())

accuracy = accuracy_score(test_y, output)
print("test accuracy is", accuracy)