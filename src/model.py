import torch
import torch.nn as nn

class RPSNet(nn.Module):
    def __init__(self, num_classes=3):
        super(RPSNet, self).__init__()
        # Based on your notebook architecture
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Add more layers here if your notebook had them
        )
        # You might need to adjust the input size for Linear layer based on your image size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 111 * 111, 128), # Example size, check your notebook printout!
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x