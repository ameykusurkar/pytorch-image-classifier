import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
EPOCHS = 20

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(F.dropout2d(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

def test():
    correct_guesses = 0

    for inputs, labels in testloader:
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        predictions = outputs.max(1, keepdim=True)[1]
        correct_guesses += predictions.eq(labels.view_as(predictions)).int().sum()

    total_inputs = len(testloader.dataset)
    print("{}/{} correct, accuracy: {}".format(int(correct_guesses), total_inputs, float(correct_guesses) / total_inputs))


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

for epoch in range(EPOCHS):
    print("EPOCH " + str(epoch))
    for data in trainloader:
        inputs, labels = data

        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

    test()
