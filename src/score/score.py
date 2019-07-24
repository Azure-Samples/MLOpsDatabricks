import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from azureml.core.model import Model
from PIL import Image


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


def init():
    global model
    model = CNN()
    model_path = Model.get_model_path(model_name="torchcnn")
    model.load_state_dict(torch.load(model_path))
    model.eval()


def run(raw_data):
    transform = transforms.transforms.Compose([
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            (0.1307,), (0.3081,))
    ])
    img = Image.frombytes(
        '1', (28, 28), (json.loads(raw_data)['data']).encode())
    input_data = transform(img)
    input_data = input_data.unsqueeze(0)
    classes = ['tshirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    output = model(input_data)
    index = torch.argmax(output, 1)
    return classes[index]
