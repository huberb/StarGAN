import torch
import torch.nn as nn
from torchsummary import summary


class Discriminator(nn.Module):

    def __init__(self, channel_dim=32, kernel=4, stride=2,
                 padding=1, device='cuda'):
        super(Discriminator, self).__init__()

        self.device = device
        self.layers = nn.Sequential(
                nn.Conv2d(3, channel_dim,
                          kernel_size=kernel, stride=stride,
                          padding=padding),
                nn.LeakyReLU(0.2, inplace=True),
                *self.block(channel_dim, channel_dim * 2,
                            kernel=kernel, stride=stride,
                            padding=padding),
                *self.block(channel_dim * 2, channel_dim * 2,
                            kernel=kernel, stride=stride,
                            padding=padding),
                *self.block(channel_dim * 2, channel_dim * 2,
                            kernel=kernel, stride=stride,
                            padding=0),
            ).to(device)
        self.class_layer = nn.Conv2d(channel_dim * 2, 1,
                                     stride=1, kernel_size=3).to(device)
        self.validity_layer = nn.Conv2d(channel_dim * 2, 1,
                                        kernel_size=3).to(device)

    def summary(self):
        summary(self, (3, 64, 64), device=self.device)

    def block(self, input_dim, output_dim, kernel, stride, padding):
        return (
            nn.Conv2d(input_dim, output_dim,
                      kernel_size=kernel, stride=stride,
                      padding=padding),
            # nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, images):
        output = self.layers(images)
        predicted_class = self.class_layer(output)
        predicted_validity = self.validity_layer(output)
        return torch.sigmoid(predicted_class).view(-1, 1), torch.sigmoid(predicted_validity).view(-1, 1)


class Generator(nn.Module):

    def __init__(self, channel_dim=64, device='cuda'):
        super(Generator, self).__init__()

        self.device = device
        self.layers = nn.Sequential(
                # input layer
                nn.Conv2d(4, channel_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
                # downsample
                nn.Conv2d(channel_dim, channel_dim * 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(channel_dim * 2),
                nn.ReLU(),
                # bottleneck
                nn.Conv2d(channel_dim * 2, channel_dim * 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channel_dim * 2),
                nn.ReLU(),
                nn.Conv2d(channel_dim * 2, channel_dim * 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channel_dim * 2),
                nn.ReLU(),
                nn.Conv2d(channel_dim * 2, channel_dim * 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channel_dim * 2),
                nn.ReLU(),
                # upscale
                nn.ConvTranspose2d(channel_dim * 2, channel_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
                # ouput layer
                nn.Conv2d(channel_dim, 3, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
                ).to(device)

    def summary(self):
        summary(self, (3, 64, 64), device=self.device)

    def forward(self, images, classes):
        classes = classes.view(classes.shape[0], classes.shape[1], 1, 1)
        classes = classes.repeat(1, 1, images.shape[2], images.shape[3])
        batch = torch.cat([images, classes], axis=1)
        return self.layers(batch)
