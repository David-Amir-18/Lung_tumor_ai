import os
import sys
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QWidget, QMessageBox, QSizePolicy
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image

class LungTumorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        with open("style.qss", "r") as file:
            style = file.read()
            self.setStyleSheet(style)

        self.setWindowTitle("Lung Tumor Detection")
        self.setGeometry(100, 100, 500, 600)

        self.label = QLabel("Lung Tumor Detector", self)
        self.label.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setFixedSize(300, 300)

        self.result = QLabel("", self)
        self.result.setAlignment(Qt.AlignCenter)

        self.upload_btn = QPushButton("Upload Image", self)
        self.upload_btn.clicked.connect(self.load_image)

        self.train_btn = QPushButton("Train Model", self)
        self.train_btn.clicked.connect(self.train_model)

        self.model_btn = QPushButton("Load Pretrained Model", self)
        self.model_btn.clicked.connect(self.load_model)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.result)
        layout.addWidget(self.upload_btn)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.model_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.create_model()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def create_model(self):
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model.to(self.device)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.pth)")
        if path:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            QMessageBox.information(self, "Model Loaded", "Pretrained model loaded successfully.")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            pixmap = QPixmap(path).scaled(300, 300, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.predict(path)

    def predict(self, path):
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = torch.sigmoid(self.model(tensor))
            prob = output.item()
            if prob >= 0.5:
                self.result.setText(f"Tumor Detected ({100*prob:.2f}% confidence)")
            else:
                self.result.setText(f"No Tumor ({100*(1 - prob):.2f}% confidence)")

    def train_model(self):
        data_dir = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if not data_dir:
            return

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(root=data_dir, transform=transform)

        # Split into 80% train and 20% validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        model = self.model
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        model.train()
        for epoch in range(3):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.float().unsqueeze(1).to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = torch.sigmoid(outputs) >= 0.5
                    correct += (preds == labels.bool()).sum().item()
                    total += labels.size(0)

            acc = correct / total
            print(f"Epoch {epoch+1}/3, Train Loss: {running_loss/len(train_loader):.4f}, Val Acc: {acc:.2%}")

        torch.save(model.state_dict(), "resnet50_lung_tumor.pth")
        QMessageBox.information(self, "Training Done", "Model trained and saved as 'resnet50_lung_tumor.pth'")
        model.eval()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LungTumorApp()
    window.show()
    sys.exit(app.exec_())
