import torch
import torch.nn as nn 
import torch.nn.functional as F 


class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[82, 82, 82, 82], interm_dim=128):
        super(LossNet, self).__init__()

        self.GAP1 = nn.AdaptiveAvgPool1d(1)
        self.GAP2 = nn.AdaptiveAvgPool1d(1)
        self.GAP3 = nn.AdaptiveAvgPool1d(1)
        

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)

        self.linear = nn.Linear(3 * interm_dim, 1)
    
    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1].permute(1, 0, 2))
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2].permute(1, 0, 2))
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out = self.linear(torch.cat((out1, out2, out3), 1))
        return out

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()