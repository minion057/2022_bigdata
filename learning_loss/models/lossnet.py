import torch
import torch.nn as nn 
import torch.nn.functional as F 


class LossNet(nn.Module):
    # def __init__(self):
    #     super(LossNet, self).__init__()
    #     self.avgpool = self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    #     self.fc1 = nn.Linear(64, 128)
    #     self.fc2 = nn.Linear(128, 128)
    #     self.fc3 = nn.Linear(256, 128)
    #     self.fc4 = nn.Linear(512, 128)
    #     self.final = nn.Linear(512, 1)

    # def forward(self, x):
    #     x[0] = self.avgpool(x[0])
    #     x[0] = torch.flatten(x[0], 1)
    #     x[0] = self.fc1(x[0])
    #     x[1] = self.avgpool(x[1])
    #     x[1] = torch.flatten(x[1], 1)
    #     x[1] = self.fc2(x[1])
    #     x[2] = self.avgpool(x[2])
    #     x[2] = torch.flatten(x[2], 1)
    #     x[2] = self.fc3(x[2])
    #     x[3] = self.avgpool(x[3])
    #     x[3] = torch.flatten(x[3], 1)
    #     x[3] = self.fc4(x[3])
    #     final = torch.cat((x[0], x[1], x[2], x[3]), dim=1)
    #     final = self.final(final) 
    #     return final

    def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[82, 82, 82, 82], interm_dim=128):
        super(LossNet, self).__init__()
        
        # self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        # self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        # self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        # self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.GAP1 = nn.AdaptiveAvgPool1d(1)
        self.GAP2 = nn.AdaptiveAvgPool1d(1)
        self.GAP3 = nn.AdaptiveAvgPool1d(1)
        

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        # self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(3 * interm_dim, 1)
    
    def forward(self, features):
        # print(features[0].size(), features[1].size(), features[2].size())
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1].permute(1, 0, 2))
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2].permute(1, 0, 2))
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        # out4 = self.GAP4(features[3])
        # out4 = out4.view(out4.size(0), -1)
        # out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3), 1))
        return out