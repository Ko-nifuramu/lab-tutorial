import torch
import torch.nn as nn


class SEblock(nn.Module):
    def __init__(self, channels, reduction_rate=4):
        super(SEblock, self).__init__()
        
        self.GAP = nn.AdaptiveAvgPool2d(1)#Global Average Pooling
        self.linear = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels//reduction_rate, bias=False),#(1, 1, channels/reduction)
            nn.ReLU(),
            nn.Linear(in_features=channels//reduction_rate, out_features=channels, bias=False),#(1, 1, channels)
            nn.Sigmoid()
        )
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        s = self.GAP(x).squeeze()
        # print(x[:, 0,0])
        s = self.linear(s).unsqueeze(-1).unsqueeze(-1)
        # print(s)
        x = torch.mul(x, s)
        # print(x[:, 0,0])
        return x

class SE_Resblock(nn.Module):
    def __init__(self, channels, bottleneck_rate=4, reduction_rate=4):
        super(SE_Resblock, self).__init__()
        
        self.neck_channels=channels//bottleneck_rate
        
        self.conv1=nn.Sequential(
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=self.neck_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=self.neck_channels),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=self.neck_channels, out_channels=self.neck_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=self.neck_channels),
            nn.ReLU()
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=self.neck_channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(num_features=channels)
        )
        
        self.relu=nn.ReLU()
        
        self.se=SEblock(channels=channels, reduction_rate=reduction_rate)
        
        
        def forward(self, x):
            res=self.conv1(x)
            # print(res.shape)
            res=self.conv2(res)
            # print(res.shape)
            res=self.conv3(res)
            # print(f'x:{x.shape}, r:{res.shape}')
            x = self.relu(x+self.se(res))
            return x
        