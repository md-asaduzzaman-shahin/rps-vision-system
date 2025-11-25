import torch
import torch.nn as nn
from torchvision import models

class RPSNet(nn.Module):
    """
    A custom Convolutional Neural Network (CNN) for Rock-Paper-Scissors classification.
    
    Architecture based on the initial experiments:
    - 1 Convolutional Block (Conv2d -> ReLU -> MaxPool2d)
    - Flattening
    - Fully Connected Classifier
    """
    def __init__(self, num_classes=3):
        super(RPSNet, self).__init__()
        
        # Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Classifier
        # Input image is 100x100.
        # After Conv2d(3x3): 98x98
        # After MaxPool2d(2x2): 49x49
        # Flatten size = 32 channels * 49 * 49 = 76832
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 49 * 49, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_transfer_model(num_classes=3, freeze_features=True):
    """
    Constructs a MobileNetV2 model for Transfer Learning.
    
    Args:
        num_classes (int): Number of output classes (default: 3).
        freeze_features (bool): If True, freezes the weights of the feature extractor.
    """
    # Load pre-trained model
    model = models.mobilenet_v2(pretrained=True)
    
    # Freeze feature extractor parameters
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
            
    # Replace the classifier head
    # MobileNetV2 last channel is 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, num_classes)
    )
    
    return model