import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import CNN  # 既に作成済みの CNN モデルを使う

# データの前処理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 0.5 平均、0.5 標準偏差で正規化
])

# MNIST データセットをダウンロード
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)

# モデルの定義
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# モデルの学習
epochs = 5  # エポック数
for epoch in range(epochs):
  model.train()
  total_loss = 0

  for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

  print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

# モデルの保存
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("モデルを mnist_cnn.pth に保存しました！")
