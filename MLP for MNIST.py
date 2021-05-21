# -*- coding: utf-8 -*-
"""
Created on Fri May 21 10:01:38 2021

@author: BZ
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:50:39 2021

@author: BZ
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

# Load the data (60 000 training images, 10 000 test images)
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)


#Create 600 mini batches with 100 images each 
batchSize = 100
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batchSize, shuffle=True) 
val_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batchSize, shuffle=True) 
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batchSize, shuffle=False)

class multiLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 500) 
        self.layer2 = nn.Linear(500,10)
        #lin is an attribute that contains the nested <Linear> model from pytorch

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return x
    

model = multiLayer()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#model.parameters() contains 10 weights, each of size 784, and 10 biases, each of size 1
# print(len(list(model.parameters())[0][0]))


numEpochs = 10
train_loss = 0
val_loss = 0
val_loss_min = np.Inf

for i in range(numEpochs):
    
    model.train()
    
    # Iterate through minibatches  (training)
    for images, labels in tqdm(train_loader):
        # Zero out the gradients
        optimizer.zero_grad()
        
        # Forward pass
        x = images.view(-1, 28*28)
        y = model(x) #prediction
        loss = criterion(y, labels)
        # Backward pass
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    # Validation
    
    with torch.no_grad():
        model.eval()
        for images, labels in tqdm(val_loader):
            
            x = images.view(-1, 28*28)
            y = model(x)
            loss = criterion(y, labels)
            val_loss += loss.item()
            
    train_loss = np.mean(train_loss)
    val_loss = np.mean(val_loss)       
    
    if val_loss < val_loss_min:
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = val_loss
            
        

## Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    # Iterate through test set minibatchs 
    for images, labels in tqdm(test_loader):
        # Forward pass
        x = images.view(-1, 28*28)
        y = model(x)
        
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())
    
print('Test accuracy: {}'.format(correct/total))

# Get weights
# fig, ax = plt.subplots(1, 10, figsize=(20, 2))
# for digit in range(10):
#     ax[digit].imshow(W[:,digit].detach().view(28,28), cmap='gray')