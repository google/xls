import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import argparse

NUM_OPCODES = 8
MAX_OP_COUNT = 8
HIDDEN_LAYER1 = 200
HIDDEN_LAYER2 = 200
NUM_EPOCHS = 40
EPSILON = 0.2
NUM_EPS = 5

class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    # Dense, 1 hidden layer
    self.fc1 = nn.Linear(NUM_OPCODES * MAX_OP_COUNT, HIDDEN_LAYER1)
    self.fc2 = nn.Linear(HIDDEN_LAYER1, HIDDEN_LAYER2)
    self.out = nn.Linear(HIDDEN_LAYER2, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)
    return x


model = Net()
criterion = nn.MSELoss()
l1 = nn.L1Loss()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)


def train(data):
  model.train()
  idxs = list(range(len(data)))
  random.shuffle(idxs)
  train_loss = 0
  for i in idxs:
    x = data[i][:-1]
    y = data[i][-1]
    x_onehot = [[1 if x[a] == j+1 else 0 for j in range(NUM_OPCODES)] for a in range(len(x))]
    x_tensor = torch.flatten(torch.FloatTensor(x_onehot))
    y_tensor = torch.FloatTensor([y])
    # Forward pass
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)
    model.zero_grad()
    # Backward pass
    loss.backward()
    # Perform step
    optimizer.step()
    train_loss += loss
  return train_loss / len(data)

def test(data, log_misses=False):
  model.eval()
  total = 0
  ep = [0 for i in range(NUM_EPS)]
  logfile = open("./data/data_{}_{}.log".format(NUM_OPCODES, MAX_OP_COUNT), "w+")

  for i in range(len(data)):
    x = data[i][:-1]
    y = data[i][-1]
    x_onehot = [[1 if x[a] == j+1 else 0 for j in range(NUM_OPCODES)] for a in range(len(x))]
    x_tensor = torch.flatten(torch.FloatTensor(x_onehot))
    y_tensor = torch.FloatTensor([y])
    y_pred = model(x_tensor)

    loss = criterion(y_pred, y_tensor)
    abs_error = l1(y_pred, y_tensor)
    for j in range(NUM_EPS):
      if abs_error < EPSILON * (j+1):
        ep[j] += 1
    total += loss
    if log_misses:
      print("X: ", data[i][:-1], file=logfile)
      print("y: ", data[i][-1], file=logfile)
      print("y_pred: ", y_pred, file=logfile)
  mse = total / len(data)
  accuracy = [x / len(data) for x in ep]
  print("Validation MSE Loss: {}".format(mse))
  print("Accuracy:", accuracy)
  logfile.close()
  return mse


def main(log_misses):
  #data = np.genfromtxt('./sample_data.csv'.format(NUM_OPCODES, MAX_OP_COUNT), delimiter=',')
  data = np.genfromtxt('./data/data_{}_{}.csv'.format(NUM_OPCODES, MAX_OP_COUNT), delimiter=',')
  train_size = int(.8*len(data))
  train_data, test_data = torch.utils.data.random_split(data, [train_size, len(data)-train_size])
  epochs = [i+1 for i in range(NUM_EPOCHS)]
  training = []
  validation = []
  log_flag = False
  for i in range(NUM_EPOCHS):
    if i == (NUM_EPOCHS - 1) and log_misses:
      log_flag = True
    print("Epoch {}".format(i))
    train_loss = train(train_data)
    valid_loss = test(test_data, log_flag)
    training.append(train_loss)
    validation.append(valid_loss)
  plt.plot(epochs, training, label='training')
  plt.plot(epochs, validation, label='validation')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.legend(loc="upper right")
  plt.savefig('./data/loss_{}_{}'.format(NUM_OPCODES, MAX_OP_COUNT))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('log_misses', type=bool)
  args = parser.parse_args()
  main(args.log_misses)
