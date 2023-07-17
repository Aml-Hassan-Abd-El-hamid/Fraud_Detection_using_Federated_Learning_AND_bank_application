import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,input_shape,out_shape=1,sig=True):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(input_shape, 500)
        self.bn1 = nn.BatchNorm1d(500)
        
        
        self.fc2 = nn.Linear(500, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        
        self.fc6 = nn.Linear(128, out_shape)
        self.dropout = nn.Dropout(0.3)
        self.sig=sig
    def forward(self, x):
        x=x.to(torch.float32)
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        
        x = torch.nn.functional.leaky_relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout(x)
        
        x = torch.nn.functional.leaky_relu(self.fc4(x))
        x = self.bn4(x)
        x = self.dropout(x)

        x = torch.nn.functional.leaky_relu(self.fc5(x))
        x = self.bn5(x)
        x = self.dropout(x)
        x=self.fc6(x)
        if self.sig:
            x = torch.sigmoid(x)
        
        return x
