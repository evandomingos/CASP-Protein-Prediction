#Various libraries
library(tidyverse)  
library(caret)
library(tensorflow) 
library(ranger) # Random forest with Ranger (more efficient than rf library for higher dimensional data)
library(randomForest) # Random Forest
library(xgboost) # For Gradient Boosted Tree
library(nnet)  # Neural net
library(keras)
library(gtable)
library(zeallot)
library(rpart)
library(rpart.plot)
library(torch)
library(luz) # High-level interface for torch
library(torchvision) # For datasets and image transformation
library(GMDHreg) # Library for Group Method of Data Handling (GMDH) Nueral Network

# Load CASP 5.9 Set from 
dataset <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv", header = TRUE, stringsAsFactors = FALSE)

# Remove missing data (none in CASP dataset), create protein set
protein <- na.omit(dataset)

#n number of protein residues: 45730 obs
n <- nrow(protein)
# For reproduction
set.seed(7)

# Number of test samples = 1/3 observations
ntest <- trunc(n / 3)
#randomly select test IDs from 
testid <- sample(1:n, ntest)

# Set PyTorch seed
torch_manual_seed(7)

# In this segment,  only Gradient Boosted tree uses
# Scale variables, separate x labels ,y target var. Remove intercepts
x <- scale(model.matrix(protein$RMSD ~ . - 1, data = protein))
length(x)
y <- protein$RMSD







# First Method
# Random Forest regression tree with the obejctive of plotting feature important and purposely overfitting a model
# Very easy to produce good looking tree, extremely overfitted and validation MAE explodes

library(randomForest)


# Split the data into training and testing sets

# *** Un-comment these blocks to run the Random Forest Model. Commented out so the train_id object does not affect Nueral Networks ***

# For reproducibility
set.seed(13)
#train_id <- sample(1:n, floor(n * 0.6), replace = FALSE)
#valid_id <- sample(setdiff(1:n, train_id), floor(n * 0.2), replace = FALSE)
#test_id <- setdiff(1:n, union(train_id, valid_id))

# Create extra holdout set, "valid"
#train_x <- x[train_id, ]
#train_y <- y[train_id]
#valid_x <- x[valid_id, ]
#valid_y <- y[valid_id]
#test_x <- x[test_id, ]
#test_y <- y[test_id]


# Train the random forest regression model
rf_model <- randomForest(train_x, train_y, ntree = 500)

# Plot feature importance
varImpPlot(rf_model, main = "Feature Importance")

# Predict RMSD for test set using the trained model
pred_valid <- predict(rf_model, valid_x)
pred_test <- predict(rf_model, test_x)

# Calculate mean squared error on the test set and validation (extra holdout data)
mae_valid <- mean(abs(pred_valid - valid_y))
mae_test <- mean(abs(pred_test-test_y))



# Scatterplot of predicted vs actual RMSD on validation set
plot(valid_y, pred_valid, xlab = "Actual RMSD (Angstrom)", ylab = "Predicted RMSD (Angstrom)", main = "RF Regression Tree")
abline(a = 0, b = 1, col = "purple")

# Scatterplot of predicted vs actual RMSD on test set
plot(test_y, pred_test, xlab = "Actual RMSD (Angstrom)", ylab = "Predicted RMSD (Angstrom)", main = "RF Regression Tree")
abline(a = 0, b = 1, col = "purple")

# MAE 2.5555 on test set 
# After purposely overfitting as much as possible (try nrounds >>500) this MAE of 2.55 appears to be the absolute minimum error achievable
mae_test

# MAE 2.541 on validation, second holdout set
mae_valid


# Conventional Regression Tree for the purpose of displaying featurew importance
protein_tree1 <- rpart(RMSD ~ ., data = protein[train_id,], control = rpart.control(minsplit = 1))

# Predict on Training set
train_pred_protein_tree1 <- predict(protein_tree1, protein[train_id,])
# Calculate MAE on training set
train_tree1_mae <- mean(abs(train_pred_protein_tree1-protein$RMSD[train_id]))
# Train MAE 5.49
train_tree1_mae

# Predict on Test Set, Calculate MAE
test_pred_protein_tree1 <- predict(protein_tree1, newdata = protein[test_id, ])

# Calculate MAE for test set
mae_protein_tree1 <- mean(abs(pred_protein_tree1 - y[testid]))
# Test MAE 5.49
mae_protein_tree1

# Plot Fitted Tree
rpart.plot(protein_tree1)
  

# Gradient Boosted Tree
# Gradient Boosted Regression Tree with MAE Objective Function
gdboosteds_protein <- xgboost(data = x[-testid,], label = y[-testid], nrounds = 10000, objective = "reg:absoluteerror")
gdboost_test_pred <- predict(gdboosteds_protein, x[testid, ])
# For set.seed(13), nrounds = 10,000, minsplit = DEFAULT, trained MAE = 2.628

# Calculate MSE
mse <- mean((gdboost_test_pred-y[testid])^2)
# 19.3254 MSE for set.seed(13), nround = 10,000
mse


# Calculate MAE on test set
mae <- mean(abs(gdboost_test_pred-y[testid]))
# 2.979239 MAE for set.seed(7), nround = 10000
mae

# Scatterplot of predicted vs actual RMSD
plot(y[testid], gdboost_test_pred, xlab = "Actual RMSD (Angstrom)", ylab = "Predicted RMSD (Angstrom)", main = "Gradient Boosted Regression Tree")
abline(a = 0, b = 1, col = "purple")


importance_matrix <- xgb.importance(model = gdboosteds_protein)
importance_matrix
xgb.plot.importance(importance_matrix, col = rgb(124/256,148/256,198/256), xlab = "Importance (0-1)", 
                    ylab = "Feature")










# Load CASP 5.9 Set from 
dataset <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv", header = TRUE, stringsAsFactors = FALSE)

# Remove missing data (none in CASP dataset), create protein set
protein <- na.omit(dataset)

# n number of protein residues: 45730 obs
n <- nrow(protein)
# For reproduction
set.seed(7)

# Number of test samples = 1/3 observations
ntest <- trunc(n / 3)
#randomly select test IDs from 
testid <- sample(1:n, ntest)

# Set PyTorch seed
torch_manual_seed(7)

# Scale variables, separate x labels ,y target var. Remove intercepts
x <- scale(model.matrix(protein$RMSD ~ . - 1, data = protein))
y <- protein$RMSD


# PYTORCH NEURAL NETWORKS

#create NN with four layers, input, 50 node hidden layer, Relu activation, dropout layer 0.4 
modnn <- nn_module(
  initialize = function(input_size) {
    self$hidden <- nn_linear(input_size, 50)
    self$activation1 <- nn_relu()
    self$dropout <- nn_dropout(0.4)
    self$output <- nn_linear(50, 1)
  },
  forward = function(x) {
    ## x %>%
    ##   self$hidden() %>%
    ##   self$activation() %>%
    ##   self$dropout() %>%
    ##   self$output() %>%
    self$output( self$dropout( self$activation1( self$hidden(x) ) ) )
  }
)

#set hyperparameters for nn_module
# MSE loss function
modnn <- set_hparams( setup(modnn ,
                            loss = nn_mse_loss(),
                            optimizer = optim_rmsprop,
                            metrics = list(luz_metric_mae())),
                      input_size = ncol(x))


#training
#takes roughly 6 hours to train at 100 epochs, 210 sec each
fitted <-   fit(modnn,
                data = list(x[-testid, ],
                            matrix(y[-testid], ncol = 1)),
                valid_data = list(x[testid, ],
                                  matrix(y[testid], ncol = 1)),
                epochs = 10)


plot(fitted)
summary(fitted)

fitted$ctx

# Predict RMSD values for test set
test_pred <- predict(fitted, x[testid, ])

# Calculate MSE on test set
mse <- mean((test_pred-y[testid])^2)
mse
# MSE of 50.7419 for 100 epoch model


# Scatterplot of predicted vs actual RMSD
plot(y[testid], test_pred, xlab = "Actual RMSD (Angstrom)", ylab = "Predicted RMSD (Angstrom)", main = "RELU HL Nueral Network")
abline(a = 0, b = 1, col = "purple")



# 2nd Model evalauted, Addition of ELU hidden layer
modnn2 <- nn_module(
  initialize = function(input_size) {
    self$hidden1 <- nn_linear(input_size, 50)
    self$activation1 <- nn_relu()
    self$dropout1 <- nn_dropout(0.4)
    self$hidden2 <- nn_linear(50, 50)
    self$activation2 <- nn_elu()
    self$dropout2 <- nn_dropout(0.4)
   # self$hidden3 <- nn_linear()
    
    self$output <- nn_linear(50, 1)
  },
  forward = function(x) {
    x %>%
      self$hidden1() %>%
      self$activation1() %>%
      self$dropout1() %>%
      self$hidden2() %>%
      self$activation2() %>%
      self$dropout2() %>%
      self$output()
  }
)

modnn2 <- set_hparams( setup(modnn2 ,
                            loss = nn_mse_loss(),
                            optimizer = optim_rmsprop,
                            metrics = list(luz_metric_mae())),
                      input_size = ncol(x))


# Training for ModNN2 
#210sec per epoch
fitted2 <-   fit(modnn2,
                data = list(x[-testid, ],
                            matrix(y[-testid], ncol = 1)),
                valid_data = list(x[testid, ],
                                  matrix(y[testid], ncol = 1)),
                epochs = 1)

plot(fitted2)
summary(fitted2)

fitted2$ctx

#predict RMSD values for test set
test_pred2 <- predict(fitted2, x[testid, ])
mae_nn2 <- mean(abs(test_pred2 - y))
mae_nn2
plot(y[testid], test_pred2, xlab = "Actual RMSD (Angstrom)", ylab = "Predicted RMSD (Angstrom)", main = "NN2: 2 Layer RELU/ELU")
abline(a = 0, b = 1, col = "purple")







#ModNN3

#add CELU hidden layer
modnn3 <- nn_module(
  initialize = function(input_size) {
    self$hidden1 <- nn_linear(input_size, 50)
    self$activation1 <- nn_relu()
    self$dropout1 <- nn_dropout(0.4)
    self$hidden2 <- nn_linear(50, 50)
    self$activation2 <- nn_elu()
    self$dropout2 <- nn_dropout(0.4)
    self$hidden3 <- nn_linear(50, 50)
    self$activation3 <- nn_celu()
    self$dropout3 <- nn_dropout(0.4)
    # self$hidden3 <- nn_linear()
    
    self$output <- nn_linear(50, 1)
  },
  forward = function(x) {
    x %>%
      self$hidden1() %>%
      self$activation1() %>%
      self$dropout1() %>%
      self$hidden2() %>%
      self$activation2() %>%
      self$dropout2() %>%
      self$hidden3() %>%
      self$activation3() %>%
      self$dropout3() %>% 
      self$output()
  }
)
nn_get_parameters(modnn3)
modnn3 <- set_hparams( setup(modnn3 ,
                             loss = nn_mse_loss(),
                             optimizer = optim_rmsprop,
                             metrics = list(luz_metric_mae())),
                       input_size = ncol(x))


# Training for ModNN3 Neural Network
# 210sec per epoch
fitted3 <-   fit(modnn3,
                 data = list(x[-testid, ],
                             matrix(y[-testid], ncol = 1)),
                 valid_data = list(x[testid, ],
                                   matrix(y[testid], ncol = 1)),
                 epochs = 100)


matrix(y[-testid], ncol = 1)

list(x[-testid, ],
     matrix(y[-testid], ncol = 1))

plot(fitted3)
summary(fitted3)
fitted3$model
fitted3$ctx



# Predict RMSD values on the test set
test_pred3 <- predict(fitted3, x[testid, ])

test_pred3 <- predict(fitted3, x[testid, ])

plot(y[testid], test_pred3, xlab = "Actual RMSD (Angstrom)", ylab = "Predicted RMSD (Angstrom)", main = "NN3: 3 Layer RELU/ELU/CELU")
abline(a = 0, b = 1, col = "purple")

# NN3 MAE calc
length(testid)



mae_nn3 <- mean(abs(test_pred3 - y[testid]))
mae_nn3 



# Addition of Binary Step (Threshold) Hidden Layer
# Threshold of 15 Angstroms
# Intended to ebtter predict RMSD hta appears to be steeped at the 15 Angstrom level
modnn4 <- nn_module(
  initialize = function(input_size) {
    self$hidden1 <- nn_linear(input_size, 50)
    self$activation1 <- nn_relu()
    self$dropout1 <- nn_dropout(0.4)
    self$hidden2 <- nn_linear(50, 50)
    self$activation2 <- nn_elu()
    self$dropout2 <- nn_dropout(0.4)
    self$hidden3 <- nn_linear(50, 50)
    self$activation3 <- nn_relu()
    #nn_threshold ( value = 15, inplace = FALSE, threshold = 15)
    self$dropout3 <- nn_dropout(0.4)
    self$hidden4 <- nn_linear(50, 50)
    self$activation4 <- nn_elu()
    self$dropout4 <- nn_dropout(0.4)
    # self$hidden3 <- nn_linear()
    
    self$output <- nn_linear(50, 1)
  },
  forward = function(x) {
    x %>%
      self$hidden1() %>%
      self$activation1() %>%
      self$dropout1() %>%
      self$hidden2() %>%
      self$activation2() %>%
      self$dropout2() %>%
      self$hidden3() %>%
      self$activation3() %>%
      self$dropout3() %>% 
      self$hidden4() %>%
      self$activation4() %>%
      self$dropout4() %>% 
      self$output()
  }
)

modnn4 <- set_hparams( setup(modnn4 ,
                             loss = nn_mse_loss(),
                             optimizer = optim_rmsprop,
                             metrics = list(luz_metric_mae())),
                       input_size = ncol(x))


# Training
# 215 seconds per epoch
fitted4 <-   fit(modnn4,
                 data = list(x[-testid, ],
                             matrix(y[-testid], ncol = 1)),
                 valid_data = list(x[testid, ],
                                   matrix(y[testid], ncol = 1)),
                 epochs = 200)

plot(fitted4)
summary(fitted4)

xfitted4$ctx

# Predict RMSD values for test set
test_pred4 <- predict(fitted4, x[testid, ])

test_pred4 <- predict(fitted4, x[testid, ])
plot(y[testid], test_pred4, xlab = "Actual RMSD (Angstrom)", ylab = "Predicted RMSD (Angstrom)", main = "3 Layer RELU/ELU Nueral Network")
abline(a = 0, b = 1, col = "purple")





# Stack on successive RELU ELU hidden layers
modnn5 <- nn_module(
  initialize = function(input_size) {
    self$hidden1 <- nn_linear(input_size, 50)
    self$activation1 <- nn_relu()
    self$dropout1 <- nn_dropout(0.5)
    
    self$hidden2 <- nn_linear(50, 50)
    self$activation2 <- nn_elu()
    self$dropout2 <- nn_dropout(0.5)
    
    self$hidden3 <- nn_linear(50, 50)
    self$activation3 <- nn_relu()
    self$dropout3 <- nn_dropout(0.5)
    
    self$hidden4 <- nn_linear(50, 50)
    self$activation4 <- nn_elu()
    self$dropout4 <- nn_dropout(0.5) 
    
    self$hidden5 <- nn_linear(50, 50)
    self$activation5 <- nn_relu()
    self$dropout5 <- nn_dropout(0.5)  
    
    self$hidden6 <- nn_linear(50, 50)
    self$activation6 <- nn_relu()
    self$dropout6 <- nn_dropout(0.5)
    
    self$hidden7 <- nn_linear(50, 50)
    self$activation7 <- nn_elu()
    self$dropout7 <- nn_dropout(0.25) 
    
    self$hidden8 <- nn_linear(50, 50)
    self$activation8 <- nn_elu()
    self$dropout8 <- nn_dropout(0.5) 
    
    self$hidden9 <- nn_linear(50, 50)
    self$activation9 <- nn_elu()
    self$dropout9 <- nn_dropout(0.5) 
    
    self$hidden10 <- nn_linear(50, 50)
    self$activation10 <- nn_relu()
    self$dropout10 <- nn_dropout(0.5) 
    
    self$hidden11 <- nn_linear(50, 50)
    self$activation11 <- nn_relu()
    self$dropout11 <- nn_dropout(0.5) 
    
    self$hidden12 <- nn_linear(50, 50)
    self$activation12 <- nn_relu()
    self$dropout12 <- nn_dropout(0.5)
    
    self$output <- nn_linear(50, 1)
  },
  forward = function(x) {
    x %>%
      self$hidden1() %>%
      self$activation1() %>%
      self$dropout1() %>%
      
      self$hidden2() %>%
      self$activation2() %>%
      self$dropout2() %>%
      
      self$hidden3() %>%
      self$activation3() %>%
      self$dropout3() %>% 
      
      self$hidden4() %>%
      self$activation4() %>%
      self$dropout4() %>%
      
      self$hidden5() %>%
      self$activation5() %>%
      self$dropout5() %>%
      
      self$hidden6() %>%
      self$activation6() %>%
      self$dropout6() %>%
      
      self$hidden7() %>%
      self$activation7() %>%
      self$dropout7() %>%
      
      self$hidden8() %>%
      self$activation8() %>%
      self$dropout8() %>%
      
      self$hidden9() %>%
      self$activation9() %>%
      self$dropout9() %>%
      
      self$hidden10() %>%
      self$activation10() %>%
      self$dropout10() %>%
      
      self$hidden11() %>%
      self$activation11() %>%
      self$dropout11() %>%
      
      self$hidden12() %>%
      self$activation12() %>%
      self$dropout12() %>%
      
      self$output()
  }
)

modnn5 <- set_hparams( setup(modnn5 ,
                             loss = nn_mse_loss(),
                             optimizer = optim_rmsprop,
                             metrics = list(luz_metric_mae())),
                       input_size = ncol(x))


#training
#210sec per epoch
fitted5 <-   fit(modnn5,
                 data = list(x[-testid, ],
                             matrix(y[-testid], ncol = 1)),
                 valid_data = list(x[testid, ],
                                   matrix(y[testid], ncol = 1)),
                 epochs = 200)

plot(fitted5)
summary(fitted5)

xfitted2$ctx

# Predict RMSD values for test set
test_pred3 <- predict(fitted3, x[testid, ])

test_pred3 <- predict(fitted3, x[testid, ])
plot(y[testid], test_pred3, xlab = "Actual RMSD (Angstrom)", ylab = "Predicted RMSD (Angstrom)")
abline(a = 0, b = 1, col = "purple")



# Separation of features Nueral Network
# Attempt to isloate Polar surface areas which appear to be important for RMSD prediction, likely a function of van der Waals force
modnn6 <- nn_module(
  initialize = function(input_size) {
    self$hidden1 <- nn_linear(input_size, 50)
    
    # Use different activation functions for each feature
    self$activation1 <- nn_relu()  # F1, F2, F5, F6, F7, F9
    self$activation2 <- nn_linear()  # F4 F3 (Polar Surface Areas)
    self$activation3 <- nn_linear()  # F8 
    
    self$dropout1 <- nn_dropout(0.4)
    self$hidden2 <- nn_linear(50, 50)
    self$dropout2 <- nn_dropout(0.4)
    
    self$output <- nn_linear(50, 1)
  },
  forward = function(x) {
    # Split the input tensor based on the feature indices
    x1 <- x[, 1:7]  # F1, F2, F3, F4, F5, F6, F7
    x2 <- x[, 8]  # F8
    x3 <- x[, 9]  # F9
    
    # Forward pass for each feature with the corresponding activation function
    out1 <- x1 %>%
      self$hidden1() %>%
      self$activation1() %>%
      self$dropout1() %>%
      self$hidden2() %>%
      self$activation2() %>%
      self$dropout2()
    
    out2 <- x2 %>%
      self$hidden1() %>%
      self$activation3()  # Use linear activation function for F8
    
    out3 <- x3 %>%
      self$hidden1() %>%
      self$activation1()  # Use ReLU activation function for F9
    
    # Concatenate the output tensors from each feature and pass through the output layer
    torch$cat(list(out1, out2, out3), dim = 2) %>%
      self$output()
  }
)

modnn6_setup <- setup(modnn6 ,
                      loss = nn_mse_loss(),
                      optimizer = optim_rmsprop,
                      metrics = list(luz_metric_mae()))

modnn6 <- set_hparams(modnn6_setup, input_size = ncol(x))

# Training ModNN6 Separation of Features Nueral Network

# Approximately 240 seconds for each epoch
fitted6 <- fit(modnn6,
               data = list(x[-testid, ],
                           matrix(y[-testid], ncol = 1)),
               valid_data = list(x[testid, ],
                                 matrix(y[testid], ncol = 1)),
               epochs = 100)






# Group Method of Data Handling Neural Network (GMDH)
# Define the base models for the GMDH NN

# GMDH function is from GMDHreg library
# Training Multilayered Iterative (MIA) GMDH Neural Network

# Fit GMDH NN model
modnn7 <- gmdh.mia(X = as.matrix(protein[-testid, 2:10]), 
                      y = as.matrix(protein[-testid, 1]), 
                      prune = ncol(protein[-testid, 2:10]), 
                      criteria = "PRESS")
# PRESS criteria: Predicted Residual Error Sum of Squares takes into account all information in data sample and it is computed without recalculation each test point.
# "test" is alternative argument, estimation of RMSE and is computationally more efficient

summary(modnn7)

# Predict on validation set for GMDH NN
pred.modnn7 <- predict(modnn7, as.matrix(protein[testid, 2:10]))

sum(is.na(pred.modnn7[,1]))
# For set.seed(13), there are 27 NA values

# Get row numbers for NA vlaues
missing_rows <- which(is.na(pred.modnn7[,1]))
missing_rows

pred.modnn7_clean <- pred.modnn7[-missing_rows, ]
# Should be 0
sum(is.na(pred.modnn7_clean))

# Remove missing rows from the test set
protein.test <- protein[testid,][-missing_rows,]
# Calculate MAE on test set
mae <- mean(abs(pred.modnn7_clean - protein.test[,1]))
# With set.seed(13), MAE is 4.359865
mae

# Plot predicted RMSD against actual RMSD
plot(protein.test[,1], pred.modnn7_clean, xlab = "Actual RMSD (Angstrom)", ylab = "Predicted RMSD (Angstrom)", main = "NN7: 136 Layer MIA GMDH NN")
abline(a = 0, b = 1, col = "purple")




# Fit GMDH GIA NN model
modnn8 <- gmdh.gia(X = as.matrix(protein[-testid, 2:10]), 
                   y = as.matrix(protein[-testid, 1]), 
                   prune = 9*2, 
                   criteria = "test")

ncol(protein[-testid, 2:10])
# Predict on validation set for GMDH NN
pred.modnn8 <- predict(modnn8, as.matrix(protein[testid, 2:10]))

sum(is.na(pred.modnn8[,1]))
# For set.seed(13), there are 27 NA values

# Get row numbers for NA vlaues
missing_rows <- which(is.na(pred.modnn7[,1]))
missing_rows

pred.modnn7_clean <- pred.modnn7[-missing_rows, ]
# Should be 0
sum(is.na(pred.modnn7_clean))

# Remove missing rows from the test set
protein.test <- protein[testid,][-missing_rows,]
# Calculate MAE on test set
mae <- mean(abs(pred.modnn7_clean - protein.test[,1]))
# With set.seed(13), MAE is 4.36
mae









