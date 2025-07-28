import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class Generator(torch.nn.Module):
    """
    Generator definition
    """
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(latent_size, 32, 4, 1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(32, 16, 5, 1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(16, 8, 5, 1, bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(8, 1, 5, 1, bias=False),
            torch.nn.Tanh()
        )
    
    def forward(self, input):
        return self.main(input)
    
class Discriminator(torch.nn.Module):
    """
    Discriminator definition
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 5, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(8, 16, 5, 1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(16, 8, 5, 1, bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(8, 1, 4, 1, bias=False),
            torch.nn.Sigmoid()    
        )
    
    def forward(self, input):
        return self.main(input)

class model_definition():
    """
    This class defines the optimiser type and the learning rate used for optimization
    """
    def __init__(self, latent_size, learning_rate_G, learning_rate_D, reg_G, reg_D, beta_1=0.5, beta_2=0.999):
        betas = (beta_1, beta_2)

        self.model_gen = Generator(latent_size)
        self.optimizerG = torch.optim.Adam(self.model_gen.parameters(), lr=learning_rate_G, betas=betas, weight_decay=reg_G)

        self.model_discr = Discriminator()
        self.optimizerD = torch.optim.Adam(self.model_discr.parameters(), lr=learning_rate_D, betas=betas, weight_decay=reg_D)

def gan(dataloader, model_save_path, image_save_path, **kwargs):
    """
    This function trains a GAN model with the given training data and hyperparameters.

    Parameters:
    x_train: Training data
    **kwargs: Additional hyperparameters
    """

    # define required hyperparameters
    lr_G = float(kwargs.get("learning_rate_G"))
    lr_D = float(kwargs.get("learning_rate_D"))
    g_iter = int(kwargs.get("g_iter"))
    d_iter = int(kwargs.get("d_iter"))
    latent_size = int(kwargs.get("latent_size"))
    reg_G = float(kwargs.get("reg_G"))
    reg_D = float(kwargs.get("reg_D"))

    model = model_definition(latent_size, lr_G, lr_D, reg_G, reg_D)

    print('Training GAN model...')
    training_gan(model, dataloader, latent_size, num_epochs=100, discr_train_iter = d_iter, gen_train_iter = g_iter)

    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))

    torch.save(model, model_save_path)

    return 


"""
training model
"""
def training_gan(model, dataloader, latent_size, num_epochs = 5, discr_train_iter = 5, gen_train_iter = 1):

    """"
    Initialize loss values for plotting
    """
    loss_plotD = np.zeros(num_epochs)
    loss_plotGD = np.zeros(num_epochs)
    loss_plotG = np.zeros(num_epochs)

    epochs = np.arange(0, num_epochs)

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(dataloader):
                
            """"" 
            Training discriminator
            """

            for i in range(0, 1):
                model.optimizerD.zero_grad()

                """"
                Calculating D(X) and loss function
                """
                outputs_1 = model.model_discr(images).view(-1, 1)
                y_train = torch.full((images.size()[0], 1), 1.0)

                """"
                Binary cross entropy loss for discriminator
                """
                loss_frm_D = torch.nn.BCELoss()(outputs_1, y_train)
                """"
                Loss of discriminator, D(x) for Wasserstein loss
                """
                # loss_frm_D = -torch.sum(outputs_1)/len(outputs_1)

                """"
                Calculating D(G(z)) and loss function
                """

                z = torch.randn(images.size()[0], latent_size, 1, 1)
                gen_output = model.model_gen(z).detach()
                outputs_2 = model.model_discr(gen_output).view(-1, 1)
                z_output = torch.full((images.size()[0], 1), 0.0)

                """"
                Binary cross entropy loss for discriminator
                """
                loss_frm_GD = torch.nn.BCELoss()(outputs_2, z_output)
                """"
                Loss of discriminator, D(G(z)) for Wasserstein loss
                """
                # loss_frm_GD = torch.sum(outputs_2)/len(outputs_2)

                '''
                Calculating Gradient Penalty term for loss function
                '''
                # eps = 0.3
                eps = torch.rand(images.shape[0], 1, 1, 1)
                eps = eps.expand_as(images)
                
                
                Z_bar = eps*images + (1 - eps)*gen_output.detach().numpy()
                Z_bar = Z_bar.requires_grad_(True)
                Z_bar_pred = model.model_discr(Z_bar).view(-1, 1)
                z_bar_grad = torch.autograd.grad(outputs=Z_bar_pred,
                                                 inputs=Z_bar,
                                                 grad_outputs=torch.ones_like(Z_bar_pred),
                                                 create_graph=True,
                                                 retain_graph=True,
                                                 only_inputs=True)[0]

                # total_loss = loss_frm_D + loss_frm_GD + 10*((z_bar_grad.norm(2, dim=1) - 1) ** 2).mean()
                total_loss = loss_frm_D + loss_frm_GD
                total_loss.backward()
                model.optimizerD.step()

            loss_plotD[epoch] = total_loss

            """
            Training generator
            """

            for j in range(0, 1):
                """
                Calculating D(G(z)) and training
                """

                model.optimizerG.zero_grad()
                z = torch.randn(images.size()[0], latent_size, 1, 1)
                outputs = model.model_discr(model.model_gen(z)).view(-1, 1)
                output_label = torch.full((images.size()[0], 1), 1.0)

                '''
                BCE loss for Generator
                '''
                loss_frm_G = torch.nn.BCELoss()(outputs, output_label)
                '''
                Loss of Generator, D(G(z)) for Wasserstein loss
                '''
                # loss_frm_G = -torch.sum(outputs)/len(outputs)
                loss_frm_G.backward()
                model.optimizerG.step()

            loss_plotG[epoch] += loss_frm_G

    """
    Plotting
    """
    print('Loss of Generator: ', loss_plotG[num_epochs - 1])
    print('Loss of Discriminator: ', loss_plotD[num_epochs - 1])

    plt.plot(epochs, loss_plotD, label='Discriminator Loss')
    # plt.plot(epochs, loss_plotGD, label='GD Loss')
    plt.plot(epochs, loss_plotG, label='Generator Loss')
    plt.tick_params(axis='both', labelsize=10)
    plt.xlabel('Iterations', fontsize='10')
    plt.ylabel('Loss', fontsize='10')
    plt.legend(fontsize='10')
    plt.grid()
    plt.savefig(r'GAN_loss_plot.png', dpi=1000)
    # plt.show()
