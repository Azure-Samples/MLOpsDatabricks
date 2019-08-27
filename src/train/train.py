import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms


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
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--AZUREML_RUN_TOKEN')
PARSER.add_argument('--AZUREML_RUN_ID')
PARSER.add_argument('--AZUREML_ARM_SUBSCRIPTION')
PARSER.add_argument('--AZUREML_ARM_RESOURCEGROUP')
PARSER.add_argument('--AZUREML_ARM_WORKSPACE_NAME')
PARSER.add_argument('--AZUREML_ARM_PROJECT_NAME')
PARSER.add_argument('--AZUREML_SCRIPT_DIRECTORY_NAME')
PARSER.add_argument('--AZUREML_RUN_TOKEN_EXPIRY')
PARSER.add_argument('--AZUREML_SERVICE_ENDPOINT')
PARSER.add_argument('--NUM_EPOCHS')
PARSER.add_argument('--MODEL_PATH')
ARGS = PARSER.parse_args()

if ARGS.MODEL_PATH == "":
    NUM_EPOCHS = 5
else:
    NUM_EPOCHS = int(ARGS.NUM_EPOCHS)

BATCH_SIZE = 100
LEARNING_RATE = 0.001
TRAIN_LOADER = torch.utils.data.DataLoader(
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

CNN_INSTANCE = CNN()

CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(CNN_INSTANCE.parameters(), lr=LEARNING_RATE)
LOSSES = []
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(TRAIN_LOADER):
        images = Variable(images.float())
        labels = Variable(labels)

        # Forward + Backward + Optimize
        OPTIMIZER.zero_grad()
        outputs = CNN_INSTANCE(images)
        loss = CRITERION(outputs, labels)
        loss.backward()
        OPTIMIZER.step()

        # LOSSES.append(loss.data[0])
        LOSSES.append(loss.data)

        if (i + 1) % 100 == 0:
            print(
                'Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'
                % (
                    epoch + 1,
                    NUM_EPOCHS,
                    i + 1,
                    len(TRAIN_LOADER) // BATCH_SIZE,
                    loss.data
                )
            )

torch.save(CNN_INSTANCE.state_dict(), ARGS.MODEL_PATH)
