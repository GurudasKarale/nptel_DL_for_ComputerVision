from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init

class Net(nn.Module):
#####Define the architecture
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 =nn.Conv2d(1,16,3,1)
        self.batchnorm1=nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.batchnorm2=nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.batchnorm3=nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7744, 128)
        self.batchnorm4=nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)

######## FORWARD PASS
    def forward(self, x):
        x = self.conv1(x)
        x=self.batchnorm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x=self.batchnorm2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x=self.batchnorm3(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x=self.batchnorm4(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

######### Train the model
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #send image, target to the device
        data, target = data.to(device), target.to(device)
        # flush out the gradients stored in optimizer
        optimizer.zero_grad()
        # pass the image to the model and assign the output to variable named output
        output = model(data)
        # calculate the loss
        loss = F.nll_loss(output, target)
        # do a backward pass
        loss.backward()
        #update the weights
        optimizer.step()

        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #send data , target to device
            data, target = data.to(device), target.to(device)
            # pass the image to the model and assign the output to variable named output
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():

######## using CPU
    device = torch.device("cpu")


####### Transforms
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

#######Load dataset
    dataset1 = datasets.MNIST('C:/Users/Mohit K/Desktop', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('C:/Users/Mohit K/Desktop', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2,batch_size=32)


    model = Net().to(device)

##########initialize weights for conv2d layer
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.ones_(m.weight.data)
            torch.nn.init.ones_(m.bias.data)

    model.apply(init_weights)

########## use Adam as an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

#scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        #scheduler.step()

######## XAVIER initialization
"""
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.zeros_(m.bias.data)
"""

if __name__ == '__main__':
    main()
