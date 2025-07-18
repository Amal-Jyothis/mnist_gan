import time
import datetime

from packages.data_collect import*
from packages.gan import*
from packages.c_gan import*
from packages.evaluation import*
from packages.img_generation import*
from packages.classifier import*

if __name__ == "__main__":
    start_time = time.time()
    print('Run strated at:', datetime.datetime.now())

    '''
    Start of input data extraction
    '''
    dataloader = input_data()

    '''
    Train gan model
    '''
    model_save_path = r'output/saved_model/cgan_model.pth'
    image_save_path = r'output/generated_images/mnist_cgan_8'
    hyperparameters = {'learning_rate_G': 1e-5,
                       'learning_rate_D': 1e-5,
                       'g_iter': 1,
                       'd_iter': 1,
                       'latent_size': 50,
                       'reg_G': 0,
                       'reg_D': 0}

    cgan(dataloader, model_save_path, image_save_path, **hyperparameters)

    image_generation(model_save_path, image_save_path, hyperparameters['latent_size'], eg_nos_latent=1000)

    # input_img_path = r'input_data/input_saved_images'
    # generated_image_path = r'output/generated_images'
    
    # print('FID Score: ', fid(generated_image_path, input_img_path))

    # classifier_model_save_path = r'output/saved_model/classifier.pth'
    # nn(dataloader, classifier_model_save_path)

    # model = torch.load(model_save_path, weights_only=False)

    # for i, (images, target) in enumerate(dataloader):
    #     target_mod_discr = target.view(-1, 1, 1, 1).repeat(1, 3, 16, 16)
    #     images = torch.cat((images, target_mod_discr), 1)
    #     val = model.model_discr(images)
    #     print(val)

    end_time = time.time()
    print('Time taken:', end_time - start_time)
    
