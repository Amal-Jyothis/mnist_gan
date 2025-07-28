import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def image_generation(path, image_save_path, latent_size, eg_nos_latent = 100):

    """
    Generating the new fake data
    """

    model = torch.load(path, weights_only=False)

    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    z = torch.randn(eg_nos_latent, latent_size, 1, 1)

    outputs_tensor = model.model_gen(z)
    outputs = outputs_tensor.detach().numpy()

    for i in range(outputs.shape[0]):
        save_path = os.path.join(image_save_path, f"generated_image_{i}.png")
        
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(outputs[i].transpose(1, 2, 0) * 0.5 + 0.5)
        plt.savefig(save_path)
        plt.close()

    return outputs_tensor