import torch

class CNN(torch.nn.Module):
    def __init__(self,in_channels,nbr_classes):
        super(CNN,self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels,out_channels=8,kernel_size=3,padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,padding=1)
        self.fc1 = torch.nn.Linear(16*7*7,nbr_classes)

    def forward(self,x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x
