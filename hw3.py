import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split


transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 2.1 Normalize the image data
])

batch_size = 512
learning_rate = 0.009
epochs = 10

cifar10_train = torchvision.datasets.CIFAR10('./', download=True,train=True, transform=transform)
cifar10_test = torchvision.datasets.CIFAR10('./',download=True, train=False, transform=transform)

# 2.1 Split dataset
train_idx, val_idx = train_test_split(list(range(len(cifar10_train))), test_size= 0.2)
train_dataset = torch.utils.data.Subset(cifar10_train, train_idx)
val_dataset = torch.utils.data.Subset(cifar10_train, val_idx)


train_loader = DataLoader(train_dataset,  batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
                nn.Linear(64 * 4 * 4, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.Linear(128, 10)
        )
        
        
    def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = x.view(x.size(0), -1)
            x = self.layer4(x)
            return x
    
model = MLP()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
train_loss, val_loss, test_losses = [], [], []
train_accuracies, val_accuracies, test_accuracies = [], [], []

for epoch in range(epochs):    
    model.train()
    current_loss = 0.0
    correct = 0
    
    for i, data in enumerate(tqdm(train_loader)):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        current_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        
    for i, data in enumerate(val_loader):
        inputs, labels = data
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        val_loss.append(loss.item())
        val_accuracies.append(100 * (torch.argmax(outputs, 1) == labels).sum().item() / batch_size)   

    train_loss.append((current_loss / len(train_loader)))
    train_accuracies.append((correct / batch_size))    
    print('Epoch: %d, Loss: %.3f' % (epoch + 1, current_loss / batch_size))
    print('Epoch: %d, Accuracy: %.3f' % (epoch + 1, correct / batch_size))
    
 
print(train_accuracies)
print('Finished training')   

# Testing
model.eval()
correct = 0                                               
total = 0                                          
testing_loss = 0.0

with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader)):
        inputs, targets = data
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()        
        test_losses.append(loss.item())
        test_accuracies.append(correct * 100 / total)
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
print('Finished testing')

# Plotting
plt.plot(train_loss, label='Training Loss', color='red')
plt.annotate('Final Training Loss: ' + str(train_loss[-1]), xy=(epochs, train_loss[-1]), xytext=(epochs, train_loss[-1] + 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.plot(test_losses, label='Testing Loss', color='blue')
plt.annotate('Final Test loss: ' + str(test_losses[-1]), xy=(epochs, test_losses[-1]), xytext=(epochs, test_losses[-1] + 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05))
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


plt.plot(test_accuracies, label='Testing Accuracy', color='purple')
plt.annotate('Final Test Accuracy: ' + str(test_accuracies[-1]), xy=(epochs, test_accuracies[-1]), xytext=(epochs, test_accuracies[-1] + 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05))
plt.plot(train_accuracies, label= 'Training Accuracy', color='green')
plt.annotate('Final Training Accuracy: ' + str(train_accuracies[-1]), xy=(epochs, train_accuracies[-1]), xytext=(epochs, train_accuracies[-1] + 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05))
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

