import os
import numpy as np
import matplotlib.pyplot as plt

plots_dir="plots"

def show_image(image):
    np_image = image.numpy()
    plt.imshow(np.transpose(np_image,(1,2,0)))
    plt.show()

def plot_losses(losses:list,losses_type:str):
    os.makedirs(plots_dir,exist_ok=True)
    save_path=os.path.join(plots_dir,f"{losses_type}.png")
    nbr_epochs=range(1,len(losses)+1)
    plt.plot(nbr_epochs,losses,'r--',label=losses_type)
    plt.title(f"{losses_type} per epochs")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(save_path)

