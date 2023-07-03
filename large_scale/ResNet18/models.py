import torch.nn as nn
from resnet import ResNet18
import tensorflow as tf

class TransResNet(nn.Module):
    def __init__(self, num_classes=10, num_dims=512, IMAGENET=False, net=ResNet18()):
        super(TransResNet, self).__init__()
        self.encoder = nn.Sequential(*list(net.children())[:-1])
        if IMAGENET:
            self.linear = list(net.children())[-1]
        else:
            self.linear = nn.Linear(num_dims, num_classes)

    def forward(self, x):
        repr = self.encoder(x)
        repr = repr.view(repr.size(0), -1)
        logits = self.linear(repr)
        return logits, repr
    
    def classify(self, repr):
        logits = self.linear(repr)
        return logits
    
def classification_model(input_dim, num_classes):
    inputs = tf.keras.layers.Input(shape=(input_dim), name='inputs')
    
    x = inputs
    
    x = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)