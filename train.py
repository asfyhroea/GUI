import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import CNN  # 既存の CNN モデルを使用

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データの前処理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNISTデータセットの読み込み
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# モデルの定義
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()  # 損失関数は CrossEntropyLoss に変更
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
epochs = 5
for epoch in range(epochs):
  model.train()
  running_loss = 0.0
  for images, labels in trainloader:
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
  print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}")

# モデルの保存
torch.save(model.state_dict(), "mnist_cnn.pth")
print("モデルを mnist_cnn.pth に保存しました！")

# テストの実行
model.eval()
correct = 0
total = 0
with torch.no_grad():
  for images, labels in testloader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    predicted = torch.argmax(outputs, dim=1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
