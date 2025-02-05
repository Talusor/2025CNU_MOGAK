from torch import nn

class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.Conv1 = nn.Sequential(
            # batch * 28 * 28 * 1
            nn.Conv2d(1, 32, kernel_size=3, padding=2),
            # batch * 30 * 30 * 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # batch * 15 * 15 * 32
        )

        self.Conv2 = nn.Sequential(
            # batch * 15 * 15 * 32
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            # batch * 17 * 17 * 64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # batch * 8 * 8 * 64
        )

        self.fc = nn.Linear(8 * 8 * 64, 10)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.dropout(x)
        x = self.Conv2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x