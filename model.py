import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(64 * 7 * 7, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 64 * 7 * 7)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x  # Softmaxは外部で適用

def load_model(path="mnist_cnn.pth"):
  model = CNN()
  model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
  model.eval()
  return model
