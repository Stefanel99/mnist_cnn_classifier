import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch,torchvision,torchmetrics
from tqdm import tqdm



batch_size=60

train_dataset = torchvision.datasets.MNIST(root="dataset/",download=True,train=True,transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_dataset = torchvision.datasets.MNIST(root="dataset/",download=True,train=False,transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)




def show_image(image):
    np_image = image.numpy()
    plt.imshow(np.transpose(np_image,(1,2,0)))
    plt.show()

data_iter = iter(train_loader)
images,labels = next(data_iter)

#show_image(torchvision.utils.make_grid(images))



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


device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN(in_channels=1,nbr_classes=10).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


#Train the CNN model
epochs = 10
for epoch in range(epochs):
    print(f"Epoch [{epoch+1}/{epochs}]")

    for batch_index, (data,targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        scores = model(data)
        loss = loss_fn(scores,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#Evaluate the CNN model
acc = torchmetrics.Accuracy(task="multiclass",num_classes=10)
precision = torchmetrics.Precision(task="multiclass",num_classes=10)
recall = torchmetrics.Recall(task="multiclass",num_classes=10)
model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        output=model(images)
        _,preds=torch.max(output,1)
        #Unpack the torch.max in order to obtain tensors, not tuples
        acc.update(preds,labels)
        precision.update(preds,labels)
        recall.update(preds,labels)
test_accuracy=acc.compute()
test_precision = precision.compute()



#Evaluate the model's performance

#fig = plt.figure()
#plt.plot(train_dataset,)

print(f"Test Accuracy: {test_accuracy}/nTest Precision: {test_precision}")