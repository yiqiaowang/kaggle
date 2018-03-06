import torch
from torch.autograd import Variable

BATCH_SIZE = 250
INPUT_DIMENSION = 64*64
HIDDEN_DIMENSION = 100
OUT_DIMENSION = 1

x = pd.read_csv("./data/head_x.csv", header=None).as_matrix()
y = pd.read_csv("./data/head_y.csv", header=None).as_matrix()


model = torch.nn.Sequential(
    torch.nn.Linear(INPUT_DIMENSION, HIDDEN_DIMENSION),
    torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN_DIMENSION, OUT_DIMENSION)
)

loss_function = torch.nn.MSELoss(size_average=False)

step_size = 1e-7
optimizer = torch.optim.Adam(model.parameters(), lr=step_size)
num_steps = 1000


for _ in range(num_steps):
    predicted_y = model(x)
    optimizer.zero_grad()

    loss.backward()

loss = loss_function(predicted_y, y)
print("eror =", loss)