import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # init the net optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, trainset, testset, epochs=10):
        self.train_loss = []
        self.test_loss = []
        for epoch in range(epochs):
            running_loss = 0
            for i, data in enumerate(trainset, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss = running_loss + loss.item()
            # append training loss to array
            self.train_loss.append(running_loss / len(trainset))
            # calculate test loss
            running_loss = 0
            for data in testset:
                inputs, labels = data
                outputs = self.forward(inputs)
                running_loss = running_loss + self.criterion(outputs, labels).item()
            # append test loss to array
            self.test_loss.append(running_loss / len(testset))
            # print losses after each epoch
            print(f'epoch {epoch}: training loss {self.train_loss[-1]}, test loss is {self.test_loss[-1]}')

    def plot_loss(self):
        plt.plot(self.train_loss, label='training loss')
        plt.plot(self.test_loss, label='test loss')
        plt.grid()
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()

    def test(self, testset):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testset:
                images, labels = data
                outputs = self.forward(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'accuracy on test set is {100 * correct / total}%')


# EOF