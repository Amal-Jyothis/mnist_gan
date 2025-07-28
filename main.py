import time
import datetime

from packages.data_collect import*
from packages.gan import*
from packages.img_generation import*

if __name__ == "__main__":
    start_time = time.time()
    print('Run started at:', datetime.datetime.now())

    '''
    Start of input data extraction
    '''
    dataloader = input_data()

    '''
    Train gan model
    '''
    model_save_path = r'output/saved_model/gan_model.pth'
    image_save_path = r'output/generated_images/'
    hyperparameters = {'learning_rate_G': 1e-4,
                       'learning_rate_D': 1e-4,
                       'g_iter': 1,
                       'd_iter': 3,
                       'latent_size': 25,
                       'reg_G': 0,
                       'reg_D': 0}

    gan(dataloader, model_save_path, image_save_path, **hyperparameters)

    '''
    Image generation
    '''
    image_generation(model_save_path, image_save_path, hyperparameters['latent_size'], eg_nos_latent=1000)

    end_time = time.time()
    print('Time taken:', end_time - start_time)
    
