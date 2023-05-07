# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from gmdhpy import gmdh

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


# ModNN Nueral Network, Single ReLU Hidden Layer
class ModNN(nn.Module):
    def __init__(self, input_size):
        super(ModNN, self).__init__()
        self.hidden = nn.Linear(input_size, 50)
        self.activation1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.output = nn.Linear(50, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation1(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

modnn = ModNN(input_size)

criterion = nn.MSELoss()
optimizer = optim.RMSprop(modnn.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = modnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()

print('Finished Training')

# Make predictions on the test set
test_pred = modnn(x_test)

# Calculation of MAE
mae = torch.abs(test_pred - y_test).mean()
print(f"MAE: {mae.item()}")

plt.scatter(y_test, test_pred)
plt.xlabel('Actual RMSD (Angstrom)')
plt.ylabel('Predicted RMSD (Angstrom)')
plt.title('Single RELU HL Nueral Network')
plt.plot([0,20], [0, 20], color='purple')
plt.show()



# Split the data into training and testing sets

# *** Un-comment these blocks to run the Random Forest Model. Commented out so the train_id object does not affect Nueral Networks ***

# For reproducibility

import random
random.seed(13)

# Split data into train, validation, and test sets
# Create extra holdout set, "valid"
train_id = random.sample(range(n), k=int(n * 0.6))
valid_id = random.sample(set(range(n)).difference(train_id), k=int(n * 0.2))
test_id = set(range(n)).difference(set(train_id).union(valid_id))

train_x, train_y = x[train_id, :], y[train_id]
valid_x, valid_y = x[valid_id, :], y[valid_id]
test_x, test_y = x[test_id, :], y[test_id]

# Train the random forest regression model
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=500)
rf_model.fit(train_x, train_y)

# Predict RMSD for test set using the trained model
pred_valid = rf_model.predict(valid_x)
pred_test = rf_model.predict(test_x)


# Calculate mean absolute error on the test set
mae_test = np.mean(np.abs(pred_test - test_y))
# MAE 2.5555 on test set
print(mae_test)

# Calculate mean absolute error on the validation set
mae_valid = np.mean(np.abs(pred_valid - valid_y))
# MAE 6.161 on test set
print(mae_valid)





# Strict Regression Tree
from sklearn.tree import DecisionTreeRegressor
protein_tree1 = DecisionTreeRegressor(min_samples_split=1)
protein_tree1.fit(protein[train_id, :-1], protein[train_id, -1])

# Predict on Training set, goal was to minimize MAE
train_pred_protein_tree1 = protein_tree1.predict(protein[train_id, :-1])
# Calculate MAE on training set
train_tree1_mae = np.mean(np.abs(train_pred_protein_tree1 - protein['RMSD'][train_id]))
print(train_tree1_mae)

# Predict on Test Set, Calculate MAE
test_pred_protein_tree1 = protein_tree1.predict(protein[test_id, :-1])

# Calculate MAE for test set
mae_protein_tree1 = np.mean(np.abs(test_pred_protein_tree1 - y[test_id]))
print(mae_protein_tree1)

# Plot decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(protein_tree1, ax=ax, feature_names=protein.columns[:-1])
plt.show()

# Gradient Boosted Tree
import xgboost as xgb
gdboosteds_protein = xgb.XGBRegressor(objective='reg:linear', n_estimators=10000)
gdboosteds_protein.fit(train_x, train_y)

gdboost_test_pred = gdboosteds_protein.predict(test_x)

# Calculate MAE for Gradient Boosted Tree
#gb_mse = np.mean((gdboost_test_pred - test_y)**2)
gb_mae = np.mean(np.abs(gdboost_test_pred - test_y))
print(gb_mae) 

# Calculate MAE
mae = np.mean(np.abs(gdboost_test_pred - test_y))
print(mae) # 2.979239 MAE for set.seed(7), nround = 10000

# Plot feature importance
xgb.plot_importance(gdboosteds_protein, color=rgb(124/256,148/256,198/256), xlabel="Importance (0-1)", ylabel="Feature")




# Fit GMDH MIA NN model
X = np.matrix(protein.loc[testid, "2":"10"])
y = np.matrix(protein.loc[testid, "1"])
modnn7 = gmdh.GMDH(prune=protein.loc[testid, "2":"10"].shape[1], criteria="PRESS")
modnn7.fit(X, y)

# Predict on validation set for GMDH NN
pred_modnn7 = modnn7.predict(np.matrix(protein.loc[testid, "2":"10"]))

np.sum(np.isnan(pred_modnn7[:, 0]))
# For set.seed(13), there are 27 NA values

# Get row numbers for NA values
missing_rows = np.where(np.isnan(pred_modnn7[:, 0]))[0]

pred_modnn7_clean = np.delete(pred_modnn7, missing_rows, axis=0)
# Should be 0
np.sum(np.isnan(pred_modnn7_clean))

# Remove missing rows from the test set
protein_test = protein.loc[testid, :].drop(missing_rows)
# Calculate MAE on test set
mae = np.mean(np.abs(pred_modnn7_clean - protein_test.loc[:, "1"]))
# With set.seed(13), MAE is 4.359865
mae

# Plot predicted RMSD against actual RMSD
import matplotlib.pyplot as plt
plt.scatter(protein_test.loc[:, "1"], pred_modnn7_clean)
plt.xlabel("Actual RMSD (Angstrom)")
plt.ylabel("Predicted RMSD (Angstrom)")
plt.title("NN8: 141 Layer MIA GMDH Neural Network")
plt.plot(np.arange(0, 16), np.arange(0, 16), color="purple")
plt.show()


# Fit GMDH GIA NN model
modnn8 = gmdh.GMDH(prune=9*2, criteria="testr")
modnn8.fit(np.matrix(protein.loc[testid, "2":"10"]), np.matrix(protein.loc[testid, "1"]))

# Predict on validation set for GMDH NN
pred_modnn8 = modnn8.predict(np.matrix(protein.loc[testid, "2":"10"]))

np.sum(np.isnan(pred_modnn8[:, 0]))
# For set.seed(13), there are 27 NA values

# Get row numbers for NA values
missing_rows = np.where(np.isnan(pred_modnn8[:, 0]))[0]

pred_modnn8_clean = np.delete(pred_modnn8, missing_rows, axis=0)
# Should be 0
np.sum(np.isnan(pred_modnn8_clean))

# Remove missing rows from the test set
protein_test = protein.loc[testid, :].drop(missing_rows)
# Calculate MAE on test set
mae = np.mean(np.abs(pred_modnn8_clean - protein_test.loc[:, "1"]))
# With set.seed(13), MAE is 4.36
mae
