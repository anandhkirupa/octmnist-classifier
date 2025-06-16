import torch
import torch.nn as nn

stride_latter = 1
class SimpleCNN(nn.Module):
    def __init__(self, input_size = 1, hidden_size1 = 256, hidden_size2 = 128, hidden_size3 = 64, hidden_size4 = 32, output_size = 4, kernel_size = 3, stride = 1, padding = 0, init_method = 'xavier', dropout = 0.4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels = input_size, out_channels = hidden_size1, kernel_size = kernel_size, stride = stride, padding = padding),
                    nn.ReLU(),
                    nn.BatchNorm2d(hidden_size1)
                )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(in_channels = hidden_size1, out_channels = hidden_size2, kernel_size = kernel_size, stride = stride, padding = padding),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2)
                )
        self.conv3 = nn.Sequential(
                    nn.Conv2d(in_channels = hidden_size2, out_channels = hidden_size3, kernel_size = kernel_size, stride = stride_latter, padding = padding),
                    nn.ReLU()           
                )
        self.conv4 = nn.Sequential(
                    nn.Conv2d(in_channels = hidden_size3, out_channels = hidden_size4, kernel_size = kernel_size, stride = stride_latter, padding = padding),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2)
                )
        self.flatten_size = hidden_size4 * 4 * 4
        self.fc1 = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(self.flatten_size, output_size),
                    nn.Dropout(dropout)
                )
        self.apply(lambda module: self._init_weights(module, init_method))

    def _init_weights(self, module, init_method):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if init_method == 'xavier':
                nn.init.xavier_uniform_(module.weight)
            elif init_method == 'kaiming':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif init_method == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc1(x)
        return x