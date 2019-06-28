import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


parser = argparse.ArgumentParser()
parser.add_argument('--AZUREML_RUN_TOKEN')
parser.add_argument('--AZUREML_RUN_ID')
parser.add_argument('--AZUREML_ARM_SUBSCRIPTION')
parser.add_argument('--AZUREML_ARM_RESOURCEGROUP')
parser.add_argument('--AZUREML_ARM_WORKSPACE_NAME')
parser.add_argument('--AZUREML_ARM_PROJECT_NAME')
parser.add_argument('--AZUREML_SCRIPT_DIRECTORY_NAME')
parser.add_argument('--AZUREML_RUN_TOKEN_EXPIRY')
parser.add_argument('--AZUREML_SERVICE_ENDPOINT')
parser.add_argument('--MODEL_PATH')
args = parser.parse_args()

num_epochs = 5
batch_size = 100
learning_rate = 0.001
train_loader = torch.utils.data.DataLoader(
    dsets.FashionMNIST(
        root='../image-learn/data/fashion',
        train=True,
        download=True,
        transform=transforms.transforms.Compose(
            [
                transforms.transforms.ToTensor(),
                transforms.transforms.Normalize(
                    (0.1307,),
                    (0.3081,)
                )
            ]
        )
    ),
    batch_size=64,
    shuffle=True
)

cnn = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
losses = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.float())
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])

        if (i+1) % 100 == 0:
            print(
                'Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'
                % (
                    epoch+1,
                    num_epochs,
                    i+1,
                    len(train_loader)//batch_size,
                    loss.data[0]
                )
            )

torch.save(cnn.state_dict(), args.MODEL_PATH)
