import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score


# BATCH_SIZE = 250
INPUT_DIMENSION = 64*64
HIDDEN_1 = 1000
HIDDEN_2 = 200
OUT_DIMENSION = 10

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


x = np.loadtxt("./data/head_x.csv", delimiter=',')
x = Variable(torch.FloatTensor(x))

y = np.loadtxt("./data/head_y.csv", delimiter=',')
y_vectors = convert_y(y)
y_vectors = Variable(torch.FloatTensor(y_vectors), requires_grad=False)



model = torch.nn.Sequential(
    torch.nn.Linear(INPUT_DIMENSION, HIDDEN_1),
    torch.nn.Sigmoid(),
    torch.nn.Linear(HIDDEN_1, HIDDEN_2),
    torch.nn.Sigmoid(),
    torch.nn.Linear(HIDDEN_2, OUT_DIMENSION)
)

loss_function = torch.nn.MSELoss(size_average=False)

step_size = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=step_size)
num_steps = 1000


for step in range(num_steps):
    predicted_y = model(x)
    loss = loss_function(predicted_y, y_vectors)
    if step % 100 == 0:
        print(step, loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

output = convert_to_label(predicted_y.data.numpy())

accuracy = accuracy_score(y, output)
print("accuracy =", accuracy)