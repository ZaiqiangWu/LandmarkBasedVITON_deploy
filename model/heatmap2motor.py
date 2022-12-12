import torch
import torch.nn as nn
import torch.nn.functional as F


class Heatmap2Motor(nn.Module):
    def __init__(self):
        super(Heatmap2Motor, self).__init__()
        self.conv1 = nn.Conv2d(25, 64,kernel_size=5,stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(62*62*32, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))#256x256X64
        x = F.relu(self.conv2(x))#128x128*128
        x = F.relu(self.conv3(x))  # 128x128*32
        #print(x.shape)
        x = x.view(-1, 62*62*32)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    pass
