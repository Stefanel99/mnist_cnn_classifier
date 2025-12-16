import torch
from model import CNN
from PIL import Image
import numpy as np
import gradio


device = "cuda" if torch.cuda.is_available() else "cpu"
model=CNN(in_channels=1,nbr_classes=10).to(device)
model.eval()

def predict(image):
    if image is None:
        return "Please draw a number !"
    if isinstance(image,dict):
        image=image['composite']
    image = Image.fromarray(image.astype('uint8')).convert('L').resize((28,28))
    image_array = np.array(image)/255.0

    image_array=1.0-image_array

    image_tensor=torch.from_numpy(image_array).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output=model(image_tensor)
        _,prediction=torch.max(output,1)
    return f"Predicted: {np.argmax(prediction.item())}"
    
    

test = gradio.Interface(
    fn=predict,
    inputs=gradio.Sketchpad(),
    outputs="text"
)


test.launch()