import os
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

def save_input_data(dataloader, save_folder_path):

    images, _  = next(iter(dataloader))
    j = 0
    
    for img in images:
        img = img.detach().numpy()
        
        save_path = os.path.join(save_folder_path, f"input_image_{j}.png")
        
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(img.transpose(1, 2, 0) * 0.5 + 0.5) 
        plt.savefig(save_path)
        plt.close()
        j = j + 1

def input_data():
    save_folder_path = 'input_data/input_saved_images/mnist'

    #Parameters
    image_size = 16
    batch_size = 64

    #image transformation details
    image_transforms = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  
            ])

    dataset = torchvision.datasets.MNIST('input_data', train=True, download=True, transform=image_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    images, _  = next(iter(dataloader))
    j = 0
    
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    for img in images:
        img = img.detach().numpy()
        
        save_path = os.path.join(save_folder_path, f"input_image_{j}.png")
        
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(img.transpose(1, 2, 0) * 0.5 + 0.5) 
        plt.savefig(save_path)
        plt.close()
        j = j + 1

    return dataloader





    

                

