#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 00:37:31 2025

@author: pavanpaj
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from Text_to_Image_GAN import G  # Your generator model
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel  # For text embeddings (example)

# Load the trained generator model
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
netG = G().to(device)
netG.load_state_dict(torch.load('./saved_models/generator_epoch_199.pth', map_location=device))
netG.eval()  # Set the model to evaluation mode

# Load a pre-trained text embedding model (e.g., BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_encoder = BertModel.from_pretrained('bert-base-uncased').to(device)


# Add a projection layer to map 768-dimensional embeddings to 1024 dimensions
class TextEmbeddingProjector(nn.Module):
    def __init__(self, input_dim=768, output_dim=1024):
        super(TextEmbeddingProjector, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.projection(x)

# Initialize the projection layer
text_projector = TextEmbeddingProjector().to(device)

# Modify the preprocess_text function to include the projection layer
def preprocess_text(text):
    """
    Convert input text into an embedding using a pre-trained text encoder and project it to the required size.
    :param text: Input text string.
    :return: Text embedding tensor.
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=64)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generate embeddings
    with torch.no_grad():
        outputs = text_encoder(**inputs)
    
    # Use the [CLS] token embedding as the text representation
    embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, 768)
    
    # Project the embedding to the required size (1024)
    embedding = text_projector(embedding)  # Shape: (batch_size, 1024)
    
    return embedding


# Function to generate an image from an embedding
def generate_image(embedding):
    """
    Generate an image from a given embedding.
    :param embedding: Text embedding tensor.
    :return: PIL image.
    """
    # Generate noise vector
    noise = torch.randn(1, 100, 1, 1, device=device)
    
    # Generate image
    with torch.no_grad():
        fake_image = netG(noise, embedding)
    
    # Convert tensor to image
    fake_image = fake_image.squeeze(0).cpu().numpy()
    fake_image = np.transpose(fake_image, (1, 2, 0))
    fake_image = (fake_image + 1) / 2  # Rescale to [0, 1]
    fake_image = (fake_image * 255).astype(np.uint8)
    
    # Display the single image using matplotlib
    plt.imshow(fake_image)  # Display the first (and only) image
    plt.axis('off')  # Hide axes
    plt.show()
    
    return Image.fromarray(fake_image)

# Main evaluation function
def eval(text):
    """
    Evaluate the model by generating an image from input text.
    :param text: Input text string.
    """
    # Generate the text embedding
    embedding = preprocess_text(text)
    
    # Generate the image
    generated_image = generate_image(embedding)
    
    # Display the image and text description
    print(f"Input Text: {text}")
    generated_image
    # generated_image.save('output_image.png')  # Save the image

# Example usage
input_text = "A bird with colorful feathers sitting on a branch"
eval(input_text)  # Generate an image for the input text