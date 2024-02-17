import torchvision
from torch import nn
from torchvision.models.resnet import ResNet101_Weights


class ResnetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)

        num_features = self.base_model.fc.in_features

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )

        self.base_model.fc = self.classifier

    def unfreeze_layers(self, unfreeze_layers: int = 50):

        for params in self.base_model.parameters():
            params.requires_grad = False

        # Get the total number of layers in the model
        total_layers = len(list(self.base_model.children()))

        # Unfreeze the last few layers
        for idx, child in enumerate(self.base_model.children()):
            if idx + 1 > total_layers - unfreeze_layers:
                for param in child.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.base_model(x)
        return x
