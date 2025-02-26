# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from data_util import Text2ImageDataset # Locally written code





# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)   # (mean, std)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0,0.02)   # (mean = 1, std)
        m.bias.data.fill_(0)  # Bias initiated with 0
  
    
# Defining the Generator
        
class G(nn.Module): # We introduce a class to define the generator.
    
    def __init__(self): # We introduce the __init__() function that will define the architecture of the generator.
        super(G, self).__init__() # We inherit from the nn.Module tools.
        
        # Text Embedding Layer
        self.text_embedding = nn.Sequential(
            nn.Linear(embed_dim, embed_out_dim),
            nn.BatchNorm1d(embed_out_dim),
            nn.LeakyReLU(0.2, inplace = True)
            )
        
        
        
        self.main = nn.Sequential( # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).
            nn.ConvTranspose2d(100 + embed_out_dim , 512, 4, 1, 0, bias= False), # We start with an inversed convolution. (in channels, out channels, kernel size, stride, padding) Bias is False as we are doing batch normalization in next step
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias= False), # We start with an inversed convolution. (in channels, out channels, kernel size, stride, padding) Bias is False as we are doing batch normalization in next step
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias= False), # We start with an inversed convolution. (in channels, out channels, kernel size, stride, padding) Bias is False as we are doing batch normalization in next step
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias= False), # We start with an inversed convolution. (in channels, out channels, kernel size, stride, padding) Bias is False as we are doing batch normalization in next step
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias= False), # We start with an inversed convolution. (in channels, out channels, kernel size, stride, padding) Bias is False as we are doing batch normalization in next step
            nn.Tanh() # We apply a Tanh rectification to break the linearity and stay between -1 and +1 which is how the discriminator will again take in the values
            )
        
    def forward(self, noise, text): # We define the forward function that takes as argument an input that will be fed to the neural network, and that will return the output containing the generated images.
        # Process Text embeddings
        text = self.text_embedding(text) # Input text embedding
        text = text.view(text.shape[0], text.shape[1], 1, 1) # Reshaping to match the dimensions of noise
        
        input = torch.cat([noise, text], 1) # Concat noise and text which will the input for main function
        output = self.main(input) # We forward propagate the signal through the whole neural network of the generator defined by self.main.
        return output # We return the output containing the generated images.
   
    
# Defining the Discriminator

class D(nn.Module): # We introduce the __init__() function that will define the architecture of the discriminator.
    
    def __init__(self): # We introduce the __init__() function that will define the architecture of the discriminator.
        super(D, self).__init__() # We inherit from the nn.Module tools.
        
        # Text Embedding Layer
        self.text_embedding = nn.Sequential(
            nn.Linear(embed_dim, embed_out_dim),
            nn.BatchNorm1d(embed_out_dim),
            nn.LeakyReLU(0.2, inplace = True)
            )
        
        self.main = nn.Sequential( # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).
            nn.Conv2d(3, 64, 4, 2, 1, bias = False), # Staring with Convolution network as the input is an image
            nn.LeakyReLU(0.2, inplace = True), #Helps gradient flow (avoids dead neurons). inplace True since we dont need a new Tensor
            nn.Conv2d(64, 128, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True), 
            nn.Conv2d(128, 256, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            )
        
        # Output Layer
        self.output = nn.Sequential(
            nn.Conv2d(512 + embed_out_dim, 1, 4, 1, 0, bias = False), # Architecture takes the text embedding at the output again to check the text
            nn.Sigmoid() # We apply a Sigmoid rectification to break the linearity and stay between 0 and 1.
            )
        
    def forward(self, image, text):
        
        # Process image features first 
        image_features = self.main(image)
        
        # Process text embedidngs
        text = self.text_embedding(text) # Input text embedding
        text = text.view(text.shape[0], text.shape[1], 1, 1) # Reshaping Text Embeddings
        text = text.repeat(1,1, image_features.shape[2], image_features.shape[3]) # Reshaped text embeddings are repeated across the spatial dimensions of the image features
        
        
        combined = torch.cat([image_features, text], 1) # Concat noise and text which will the input for main function
        output = self.output(combined)
        return output.view(-1)


# Hyperparamaters defining
batchSize = 256
imageSize = 64
embed_dim = 1024 # Dimension of text embeddings
embed_out_dim = 128 # Output Dimension of text embeddings after processing
l1_coef = 50  # Weight for L1 loss
l2_coef = 100  # Weight for L2 loss

epochs = 1
log_interval = 27 # Define log_interval (e.g., log progress every 54 batches)



if __name__ == "__main__":
    
    # Directory setup
    output_save_path = './generated_images/'
    model_save_path = './saved_models/'
    os.makedirs(output_save_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)


    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # If using Nvidia GPU
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu") # If using Mac GPU
    print("Using Device:", device)

    #dataset = dset.CIFAR10(root = "./data", download= True, transform=transform)
    dataset = Text2ImageDataset('data/birds.hdf5',split=0) # split { 0: train, 1: validation, 2: test }
    """
    To get the birds.hdf5 data:
       : First download the data birds data from here https://drive.google.com/file/d/0B0ywwgffWnLLZW9uVHNjb2JmNlE/view?resourcekey=0-8y2UVmBHAlG26HafWYNoFQ
       : In the config.yaml path change the path accordingly
       : run convert_cub_to_hdf5_script.py script which should output the hdf5 file

    Hd5 file taxonomy `

    split (train | valid | test )
    example_name
    'name'
    'img'
    'embeddings'
    'class'
    'txt'

    """

    # Using dataLoader to get the images of the training set batch by batch.
    # A higher num_workers (>1) enables faster data loading by using multiple CPU cores.

    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batchSize, shuffle = True, num_workers=0, pin_memory = True)
    print("No of batches: ",len(dataloader))

    # Print a Sample Input
    sample = dataset[0]  # Get the first item from the dataset
    print("Sample Keys:", sample.keys())

    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: Tensor of shape {value.shape}")
        elif isinstance(value, str):
            print(f"{key}: {value}")
        elif isinstance(value, int):
            print(f"{key}: {value}")
        elif isinstance(value, np.ndarray):
            print(f"{key}: Numpy array of shape {value.shape}")
        else:
            print(f"{key}: {type(value)}")
    
    # Creating the Generator
    netG = G().to(device) # We create the generator object.
    netG.apply(weights_init) # We initialize all the weights of its neural network.

    # Creating the Discriminator'
    netD = D().to(device)
    netD.apply(weights_init)

    # Training the DC-GANs

    # Loss functions
    criterion = nn.BCELoss() # We create a Binary Cross Entropy criterion object that will measure the error between the prediction and the target.
    l1_loss = nn.L1Loss() # L1 Loss (Pixel-wise) Improves clarity
    l2_loss = nn.MSELoss() # L2 Loss (Feature-Wise) Improves Features described in the image

    #Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999)) # We create the optimizer object of the discriminator.
    optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999)) # We create the optimizer object of the Generator.

    # Lists to store losses
    D_losses = []
    G_losses = []

    # Training loop
    start_time = time.time()

    for epoch in range(epochs):

        batch_time = time.time()  # Start time for the epoch
        for i, batch in enumerate(dataloader): # iterte over the images of the dataset

            # Load real images and their text embeddings
            # We get a real image, wrong image  and the respective text embedding of the dataset which will be used to train the discriminator.
            real_images = batch['right_images'].to(device) 
            wrong_images = batch['wrong_images'].to(device)
            text_embeddings = batch['right_embed'].to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(real_images.size(0), device = device)
            fake_labels = torch.zeros(real_images.size(0), device = device)

            # 1st Step : Updating the weights of the Discriminator and Generator
            netD.zero_grad() # Important : We initialize to 0 the gradients of the discriminator with respect to the weights.

            # Train the Discriminator with some real images of the dataset
            output_real = netD(real_images, text_embeddings) # We forward propagate this real image into the neural network of the discriminator to get the prediction (a value between 0 and 1).
            errD_real = criterion(output_real, real_labels) # We compute the loss between the predictions (output) and the target (equal to 1).

            # Train the Discriminator with mismatched text image pairs so it knows wrong ones
            output_wrong = netD(wrong_images, text_embeddings) # We forward propagate this wrong image into the neural network of the discriminator to get the prediction (a value between 0 and 1).
            errD_wrong = criterion(output_wrong, fake_labels) # We compute the loss between the predictions (output) and the target (equal to 0).

            # Train the Discriminator with some fake images 
            noise = torch.randn(real_images.size(0), 100, 1, 1, device = device) # We make a random input vector (noise - normal distributed) [called latent vector] of the generator.
            fake_images = netG(noise, text_embeddings) # We forward propagate this noise and text embeddings into the neural network of the generator to get some fake generated images.
            output_fake = netD(fake_images.detach(), text_embeddings) # We forward propagate the fake generated images and the text embedding into the neural network of the discriminator to get the prediction (a value between 0 and 1).
            errD_fake = criterion(output_fake, fake_labels) # We compute the loss between the prediction (output) and the target (equal to 0).

            # Backpropagating the total error
            errD = errD_real + errD_wrong + errD_fake
            errD.backward() # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the discriminator.
            optimizerD.step() # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the discriminator.

            # 2nd Step: Updating the weights of the neural network of the generator
            netG.zero_grad()  # Important : We initialize to 0 the gradients of the generator with respect to the weights.

            output = netD(fake_images, text_embeddings) # We forward propagate the fake generated images and text images into the neural network of the discriminator to get the prediction (a value between 0 and 1).
            errG_bce = criterion(output, real_labels) # We compute the loss between the prediction (output between 0 and 1) and the target (equal to 1).

            errG_l1 = l1_coef * l1_loss(fake_images, real_images)
            errG_l2 = l2_coef * l2_loss(fake_images, real_images)

            errG = errG_bce + errG_l1 + errG_l2

            errG.backward() # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the generator.
            optimizerG.step() # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the generator.

            # Store losses
            D_losses.append(errD.item())
            G_losses.append(errG.item())

            # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps
            # Progress based on log_interval
            if (i + 1) % log_interval == 0 and i > 0:
                print('Epoch {} [{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} Time: {:.2f}'.format(
                    epoch + 1, i + 1, len(dataloader),
                    errD.mean().item(),
                    errG.mean().item(),
                    time.time() - batch_time))

            # Save generator output after every 10 epochs
            if i == len(dataloader) - 1 and ((epoch + 1) % 10 == 0 or epoch == 0):
                viz_sample = torch.cat((real_images[:32], fake_images[:32]), 0)
                vutils.save_image(viz_sample,
                                  os.path.join(output_save_path, 'output_epoch_{:03d}.png'.format(epoch + 1)),
                                  nrow=8, normalize=True)

                # Save models for every 10 epochs
                torch.save(netG.state_dict(), os.path.join(model_save_path, 'generator_epoch_%03d.pth' % epoch))
                torch.save(netD.state_dict(), os.path.join(model_save_path, 'discriminator_epoch_%03d.pth' % epoch))

    # Save final models
    torch.save(netG.state_dict(), os.path.join(model_save_path, 'generator_final.pth'))
    torch.save(netD.state_dict(), os.path.join(model_save_path, 'discriminator_final.pth'))

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_save_path, 'loss_plot.png'))
    plt.show()

    print('Total time for Execution: {}'.format(time.time()-start_time))





