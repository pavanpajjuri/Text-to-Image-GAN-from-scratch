import torch
import clip
import numpy as np
from PIL import Image
from Text_to_Image_GAN import G  # Your generator model
import torch.nn as nn
import matplotlib.pyplot as plt

# Load the trained generator model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = G().to(device)
netG.load_state_dict(torch.load('./saved_models/generator_epoch_199.pth', map_location=device))
netG.eval()  # Set to evaluation mode

# Load CLIP model for text encoding
clip_model, _ = clip.load("ViT-B/32", device=device)

# Projection layer: Convert 512-dimensional CLIP embeddings → 1024-dimensional embeddings
class ClipTextProjector(nn.Module):
    def __init__(self, input_dim=512, output_dim=1024):
        super(ClipTextProjector, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# Initialize the projection layer
text_projector = ClipTextProjector().to(device)

def preprocess_text_clip(text):
    """
    Convert input text into an embedding using CLIP's text encoder.
    :param text: Input text string.
    :return: Text embedding tensor (projected to 1024 dimensions).
    """
    # Tokenize and encode using CLIP
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_tokens)  # Output: (1, 512)

    # Project embedding to 1024 dimensions
    text_embedding = text_projector(text_embedding)  # Output: (1, 1024)

    return text_embedding  # Shape: (1, 1024)

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
    plt.imshow(fake_image)
    plt.axis('off')
    plt.show()
    
    return Image.fromarray(fake_image)

# Main evaluation function using CLIP
def eval(text):
    """
    Evaluate the model by generating an image from input text using CLIP encoding.
    :param text: Input text string.
    """
    # Generate the text embedding using CLIP
    embedding = preprocess_text_clip(text)
    
    # Generate the image
    generated_image = generate_image(embedding)
    
    # Display the image and text description
    print(f"Input Text: {text}")
    return generated_image

# Example usage
input_text = "A bird with colorful feathers sitting on a branch"
eval(input_text)  # Generate an image for the input text
