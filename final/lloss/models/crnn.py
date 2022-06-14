import torch
import torch.nn as nn 

class Bidirectional(nn.Module):
    def __init__(self, inp, hidden, out, lstm=True):
        super(Bidirectional, self).__init__()
        if lstm:
            self.rnn = nn.LSTM(inp, hidden, bidirectional=True)
        else:
            self.rnn = nn.GRU(inp, hidden, bidirectional=True)
        self.embedding = nn.Linear(hidden*2, out)
    def forward(self, X):
        recurrent, _ = self.rnn(X)
        out = self.embedding(recurrent)     
        return out


class CRNN(nn.Module):
    def __init__(self, in_channels, output):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 256, 9, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(3, 3),
                nn.Conv2d(256, 256, 4, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256))
                # Flatten 추가해보기
        
        self.linear = nn.Linear(20992, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.rnn = Bidirectional(256, 1024, output+1)

    def forward(self, X, y=None, criterion = None):
        # print(X.size())
        out = self.cnn(X)
        # print(out.size())
        N, C, w, h = out.size()
        out = out.view(N, -1, h)
        # print(out.size())
        out = out.permute(0, 2, 1)
        # print(out.size())
        out2 = self.linear(out)
        # print(out.size())

        out2 = out2.permute(1, 0, 2)
        out3 = self.rnn(out2)
            
        if y is not None:
            T = out3.size(0)
            N = out3.size(1)
        
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
            target_lengths = torch.full(size=(N,), fill_value=5, dtype=torch.int32)
        
            # out3 = out3.log_softmax(-1)
            loss = criterion(out3, y, input_lengths, target_lengths)
            
            return out3, [out, out2, out3], loss
        
        return out3, [out, out2, out3], None
    
    def _ConvLayer(self, inp, out, kernel, stride, padding, bn=False):
        if bn:
            conv = [
                nn.Conv2d(inp, out, kernel, stride=stride, padding=padding),
                nn.ReLU(),
                nn.BatchNorm2d(out)
            ]
        else:
            conv = [
                nn.Conv2d(inp, out, kernel, stride=stride, padding=padding),
                nn.ReLU()
            ]
        return nn.Sequential(*conv)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
            elif isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()