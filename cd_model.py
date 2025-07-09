import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
class Resnet18Model(nn.Module):
    def __init__(self):
        super(Resnet18Model, self).__init__()
        self.model = nn.Sequential(
            resnet18(weights=ResNet18_Weights.DEFAULT),
            nn.Linear(1000,2),
        )

    def forward(self, X):
        logits = self.model(X)
        probs = nn.functional.softmax(logits)
        return logits, probs

class Resnet50Model(nn.Module):
    def __init__(self):
        super(Resnet50Model, self).__init__()
        self.model = nn.Sequential(
            resnet50(weights=ResNet50_Weights.DEFAULT), # Pretrained weights
            # resnet50(), # Model performs much worse without pretrained weights
            nn.Linear(1000,2),
        )

    def forward(self, X):
        logits = self.model(X)
        probs = nn.functional.softmax(logits)
        return logits, probs