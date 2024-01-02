from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)  # 'same' padding
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.4)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)  # 'same' padding
        self.bn6 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.4)

        self.conv7 = nn.Conv2d(64, 128, kernel_size=4, stride=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.dropout3 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128, 10)  # Assuming input image size 28x28 after convolutions
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = nn.ReLU()(self.bn3(self.conv3(x)))
        x = self.dropout1(x)

        x = nn.ReLU()(self.bn4(self.conv4(x)))
        x = nn.ReLU()(self.bn5(self.conv5(x)))
        x = nn.ReLU()(self.bn6(self.conv6(x)))
        x = self.dropout2(x)

        x = nn.ReLU()(self.bn7(self.conv7(x)))
        x = self.flatten(x)
        x = self.dropout3(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x