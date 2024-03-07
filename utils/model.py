"""
Adapted from https://github.com/jingpingjiao/siamese_cnn/tree/master
"""

import mlx.nn as nn


def get_siamese_model(model_num):
    if model_num == 1:
        model = SiameseNetwork1()
    elif model_num == 3:
        model = SiameseNetwork3()
    elif model_num == 4:
        model = SiameseNetwork4()
    elif model_num == 6:
        model = SiameseNetwork6()
    else:
        model = SiameseNetwork1()
    return model


class Flatten(nn.Module):
    def __call__(self, input):
        # Flatten the input tensor
        return input.reshape(input.shape[0], -1)


class SiameseNetwork1(nn.Module):
    def __init__(self):
        super(SiameseNetwork1, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm(512),

            Flatten(),
            nn.Linear(320000, 1024),
            nn.ReLU(),
            nn.BatchNorm(1024)
        )

        # NOTE: Not used but trained models with this defined so it needs to stay here for loading trained weights
        self.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.cnn(x)
        return output

    def __call__(self, input1, input2=None):
        output1 = self.forward(input1)
        if input2 is None:
            return output1

        output2 = self.forward(input2)
        return output1, output2


class SiameseNetwork3(nn.Module):
    def __init__(self):
        super(SiameseNetwork3, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),

            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm(512),

            Flatten(),
            nn.Linear(320000, 1024),
            nn.ReLU(),
            nn.BatchNorm(1024)
        )

        # NOTE: Not used but trained models with this defined so it needs to stay here for loading trained weights
        self.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.cnn(x)
        return output

    def __call__(self, input1, input2=None):
        output1 = self.forward(input1)
        if input2 is None:
            return output1

        output2 = self.forward(input2)
        return output1, output2


class SiameseNetwork4(nn.Module):
    def __init__(self):
        super(SiameseNetwork4, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),

            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm(512),
            nn.ReLU(),

            Flatten(),
            nn.Linear(320000, 1024),
            nn.ReLU(),
            nn.BatchNorm(1024)
        )

    def forward(self, x):
        output = self.cnn(x)
        return output

    def __call__(self, input1, input2=None):
        output1 = self.forward(input1)
        if input2 is None:
            return output1

        output2 = self.forward(input2)
        return output1, output2

class SiameseNetwork6(nn.Module):
    def __init__(self):
        super(SiameseNetwork6, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.adaptive_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc = nn.Sequential(
            # Fully Connected Layer
            Flatten(),
            nn.Linear(18432, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm(1024),
        )

    def forward(self, x):
        return self.fc(self.adaptive_pool(self.cnn(x)))

    def __call__(self, input1, input2=None):
        output1 = self.forward(input1)
        if input2 is None:
            return output1

        output2 = self.forward(input2)
        return output1, output2