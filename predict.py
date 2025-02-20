import sys
import torch
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout
from PySide6.QtGui import QPainter, QPen, QPixmap, QImage
from PySide6.QtCore import Qt, QPoint
from model import load_model
import torchvision.transforms as transforms
from PIL import Image

class PaintApp(QWidget):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("手書き数字分類・精度")
    self.setGeometry(100, 100, 320, 400)

    self.canvas = QPixmap(280, 280)
    self.canvas.fill(Qt.GlobalColor.white)
    self.drawing = False
    self.last_point = QPoint()

    self.model = load_model()

    self.initUI()

  def initUI(self):
    layout = QVBoxLayout()

    self.label = QLabel("Draw a digit:")
    layout.addWidget(self.label)

    self.clear_button = QPushButton("消す")
    self.clear_button.clicked.connect(self.clearCanvas)
    layout.addWidget(self.clear_button)

    self.predict_button = QPushButton("決定")
    self.predict_button.clicked.connect(self.predictDigit)
    layout.addWidget(self.predict_button)

    self.setLayout(layout)

  def paintEvent(self, event):
    painter = QPainter(self)
    painter.drawPixmap(20, 20, self.canvas)

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
      painter.drawLine(self.last_point - QPoint(20, 20),
                       event.pos() - QPoint(20, 20))
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
    image = image.convertToFormat(QImage.Format_Grayscale8)
    buffer = image.bits().tobytes()
    img = np.frombuffer(buffer, dtype=np.uint8).reshape((280, 280))

    img = Image.fromarray(img).resize((28, 28))
    img = transforms.ToTensor()(img).unsqueeze(0)

    with torch.no_grad():
      output = self.model(img)
      prediction = torch.argmax(output, dim=1).item()
      confidence = torch.nn.functional.softmax(
          output, dim=1)[0, prediction].item() * 100

    self.label.setText(f"Prediction: {prediction} ({confidence:.2f}%)")

if __name__ == "__main__":
  app = QApplication(sys.argv)
  window = PaintApp()
  window.show()
  sys.exit(app.exec())
