# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Load CASP 5.9 Set
dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv", header=0)

# Remove missing data (none in CASP dataset), create protein set
protein = dataset.dropna()

# Set torch seed
torch.manual_seed(13)

# Scale variables, separate x labels, y target var. Remove intercepts
x = scale(protein.drop(['RMSD'], axis=1))
y = protein['RMSD']

# Create train and test datasets
ntest = int(len(protein) / 3)
testid = np.random.choice(np.arange(len(protein)), ntest, replace=False)

x_train = torch.tensor(x[np.delete(np.arange(len(protein)), testid)], dtype=torch.float)
y_train = torch.tensor(y[np.delete(np.arange(len(protein)), testid)], dtype=torch.float)
x_test = torch.tensor(x[testid], dtype=torch.float)
y_test = torch.tensor(y[testid], dtype=torch.float)

# Create neural network with four layers
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(x.shape[1], 50)
        self.activation1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.output = nn.Linear(50, 1)

    def forward(self, x):
        x = self.output(self.dropout(self.activation1(self.hidden(x))))
        return x

# Define the model
model = Net()

# Set hyperparameters for the neural network
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters())
trainloader = DataLoader(TensorDataset(x_train, y_train), batch_size=32)
testloader = DataLoader(TensorDataset(x_test, y_test), batch_size=32)

# Train the neural network
epochs = 100
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    with torch.no_grad():
        test_outputs = model(x_test)
        test_loss = criterion(test_outputs, y_test.view(-1, 1))

    print('[%d, %5d] loss: %.3f test_loss: %.3f' %
          (epoch + 1, i + 1, running_loss / len(trainloader), test_loss.item()))

# Predict RMSD values for test set
test_pred = model(x_test)
mse = criterion(test_pred, y_test.view(-1, 1))
print('MSE: ', mse.item())

# Plot predicted vs actual RMSD
import matplotlib.pyplot as plt
plt.scatter(y_test, test_pred.detach().numpy())
plt.plot(y_test, y_test, color='purple')
plt.xlabel('Actual RMSD (Angstrom)')
plt.ylabel('Predicted RMSD (Angstrom)')
plt.title('RELU HL Nueral Network')
plt.show()



