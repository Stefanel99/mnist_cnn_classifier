import os
import torch
import torchvision
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from model import CNN

batch_size = 60
plots_dir = "plots"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

test_dataset = torchvision.datasets.MNIST(root="dataset/",download=True,train=False,transform=torchvision.transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


model = CNN(in_channels=1, nbr_classes=10).to(device)
model.load_state_dict(torch.load("mnist_cnn_model.pth", map_location=device))
model.eval()

print("Model loaded successfully!")

acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
precision = torchmetrics.Precision(task="multiclass", num_classes=10)
recall = torchmetrics.Recall(task="multiclass", num_classes=10)

all_preds = []
all_labels = []


with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        _, preds = torch.max(output, 1)
        
        acc.update(preds.cpu(), labels.cpu())
        precision.update(preds.cpu(), labels.cpu())
        recall.update(preds.cpu(), labels.cpu())
        
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

test_accuracy = acc.compute()
test_precision = precision.compute()
test_recall = recall.compute()

print(f"\nTest Results:")
print(f"Accuracy: {round(float(test_accuracy), 3) * 100}%")
print(f"Precision: {round(float(test_precision), 3) * 100}%")
print(f"Recall: {round(float(test_recall), 3) * 100}%")

cm = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=10)
confusion_matrix = cm(all_preds, all_labels)
converted_confusion_matrix = confusion_matrix.cpu().numpy()

plt.figure(figsize=(10, 8))
sns.heatmap(
    converted_confusion_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=range(10),
    yticklabels=range(10)
)

plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
os.makedirs(plots_dir, exist_ok=True)
save_path = os.path.join(plots_dir, "confusion_matrix.png")
plt.savefig(save_path)
print(f"\nConfusion matrix saved to {save_path}")