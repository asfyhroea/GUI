import sys
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
from model import load_model
import torchvision.transforms as transforms
from PIL import Image

class PaintApp(QWidget):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("Handwritten Digit Recognition")
    self.setGeometry(100, 100, 300, 350)
    self.canvas = QPixmap(280, 280)
    self.canvas.fill(Qt.GlobalColor.white)
    self.drawing = False
    self.last_point = QPoint()
    self.initUI()
    self.model = load_model()

  def initUI(self):
    self.label = QLabel("Draw a digit:", self)
    self.label.setGeometry(10, 290, 150, 30)

    self.clear_button = QPushButton("Clear", self)
    self.clear_button.setGeometry(200, 290, 80, 30)
    self.clear_button.clicked.connect(self.clearCanvas)

    self.predict_button = QPushButton("Predict", self)
    self.predict_button.setGeometry(100, 290, 80, 30)
    self.predict_button.clicked.connect(self.predictDigit)

  def paintEvent(self, event):
    painter = QPainter(self)
    painter.drawPixmap(10, 10, self.canvas)

  def mousePressEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton:
      self.drawing = True
      self.last_point = event.pos()

  def mouseMoveEvent(self, event):
    if self.drawing:
      painter = QPainter(self.canvas)
      pen = QPen(Qt.GlobalColor.black, 12, Qt.PenStyle.SolidLine,
                 Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
      painter.setPen(pen)
      painter.drawLine(self.last_point - QPoint(10, 10),
                       event.pos() - QPoint(10, 10))
      self.last_point = event.pos()
      self.update()

  def mouseReleaseEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton:
      self.drawing = False

  def clearCanvas(self):
    self.canvas.fill(Qt.GlobalColor.white)
    self.update()

  def predictDigit(self):
    image = self.canvas.toImage()
    image = image.convertToFormat(QImage.Format_Grayscale8)  # 修正

    # QImageのデータをnumpy配列に変換
    buffer = image.bits().tobytes()  # 修正
    img = np.frombuffer(buffer, dtype=np.uint8).reshape((280, 280))

    img = Image.fromarray(img)
    img = img.resize((28, 28))
    img = transforms.ToTensor()(img).unsqueeze(0)

    with torch.no_grad():
      output = self.model(img)
      probs = output.numpy().flatten()
      prediction = np.argmax(probs)
      confidence = probs[prediction] * 100

    self.label.setText(f"Prediction: {prediction} ({confidence:.2f}%)")

if __name__ == "__main__":
  app = QApplication(sys.argv)
  window = PaintApp()
  window.show()
  sys.exit(app.exec_())
