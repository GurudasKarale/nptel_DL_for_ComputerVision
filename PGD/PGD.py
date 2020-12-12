from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

epsilons = 0.3
pretrained_model = "C:/Users/Mohit K/Downloads/lenet_mnist_model.pth"
use_cuda = True


# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
        # return F.log_softmax(x, dim=1)


# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()


## Implement Projected Gradient Descent algorithm

## YOUR CODE STARTS HERE
def PGD_attack(X, y, model):
    images = X.to(device)
    labels =y.to(device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(40):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + 0.007843137 * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-0.3, max=0.3)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images



    """
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(40):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + X.shape[0] * 0.007843137 * delta.grad.data).clamp(-0.3, 0.3)
        delta.grad.zero_()
    return delta.detach()
    #return adv_image
    """

### YOUR CODE ENDS HERE

def test(model, device, data, target, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Send the data and label to the device
    data, target = data.to(device), target.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    # Forward pass the data through the model
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

    ### generate the perturbed image using PGD
    perturbed_data = PGD_attack(data, target, model)

    # Re-classify the perturbed image
    output = model(perturbed_data)

    # Check for success
    final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    if final_pred.item() == target.item():
        correct += 1
    else:
        pass
    # Return the accuracy and an adversarial example
    return final_pred, perturbed_data


for data, target, in test_loader:
    data = data[0:1, :, :, :]
    target = target[0:1]
    break

pred_adv, adv_ex = test(model, device, data, target, epsilons)
print("Predicted class for perturbed image: ", pred_adv)

# Compute the mean pixel value of the adversarial image
import tensorflow as tf

g=tf.reshape(adv_ex,[-1])

p=tf.reshape(g,[28,28])

print(np.mean(p))



