import torch.nn as nn
import torchvision.models as models


class TransResNet(nn.Module):
    def __init__(self, num_classes=10, IMAGENET=False, net=models.resnet18(), clip=False):
        super(TransResNet, self).__init__()
        self.clip = clip
        if clip:
            self.encoder = net
            num_dims=512
        else:
            self.encoder = nn.Sequential(*list(net.children())[:-1])
            num_dims = list(net.children())[-1].in_features
        if IMAGENET and not clip:
            self.linear = list(net.children())[-1]
        else:
            self.linear = nn.Linear(num_dims, num_classes)

    def forward(self, x):
        if self.clip:
            repr = self.encoder.encode_image(x)
        else:
            repr = self.encoder(x)
        repr = repr.view(repr.size(0), -1)
        logits = self.linear(repr)
        return logits, repr
    
    def classify(self, repr):
        logits = self.linear(repr)
        return logits
    