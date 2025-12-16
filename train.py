import torch
import torchvision
import torchmetrics
from tqdm import tqdm
from model import CNN
from utils import plot_losses


batch_size=60
epochs = 10
learning_rate=0.001

train_dataset = torchvision.datasets.MNIST(root="dataset/",download=True,train=True,transform=torchvision.transforms.ToTensor())

train_size=int(.8*len(train_dataset))
value_size=len(train_dataset)-train_size


train_subset,value_subset=torch.utils.data.random_split(train_dataset,[train_size,value_size])

train_loader = torch.utils.data.DataLoader(dataset=train_subset,batch_size=batch_size,shuffle=True)
value_loader = torch.utils.data.DataLoader(dataset=value_subset,batch_size=batch_size,shuffle=True)

test_dataset = torchvision.datasets.MNIST(root="dataset/",download=True,train=False,transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)


data_iter = iter(train_loader)
images,labels = next(data_iter)



device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN(in_channels=1,nbr_classes=10).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


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



acc = torchmetrics.Accuracy(task="multiclass",num_classes=10)
precision = torchmetrics.Precision(task="multiclass",num_classes=10)
recall = torchmetrics.Recall(task="multiclass",num_classes=10)


epochs = 10

train_losses=[]
val_losses=[]

for epoch in range(epochs):
    print(f"Epoch [{epoch+1}/{epochs}]")

    model.train()
    batch_losses=[]

    for batch_index, (data,targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        scores = model(data)
        loss = loss_fn(scores,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
    train_losses.append(sum(batch_losses)/len(batch_losses))

    model.eval()
    val_batch_loss=[]
    all_preds=[]
    all_labels=[]
    with torch.no_grad():
        for images, labels in value_loader:
            output=model(images)
            _,preds=torch.max(output,1)
            acc.update(preds,labels)
            precision.update(preds,labels)
            recall.update(preds,labels)
            val_loss=loss_fn(output,labels)
            val_batch_loss.append(val_loss.item())

    val_losses.append(sum(val_batch_loss)/len(val_batch_loss))


test_accuracy=acc.compute()

print(f"Test Accuracy: {round(float(test_accuracy),3)*100}%")


torch.save(model.state_dict(),"mnist_cnn_model.pth")
print("\nModel saved to mnist_cnn_model.pth")


plot_losses(train_losses,'Training Loss')
plot_losses(val_losses,'Validation Loss')

print("\nTraining complete !")