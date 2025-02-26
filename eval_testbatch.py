#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Evaluation Script with IS, FID, CLIP Similarity, and Grid Visualization

You can change the number of images to evaluate by modifying the `num_images` variable.
"""

import torch
import numpy as np
from PIL import Image
import h5py  # For loading the CUB dataset
from Text_to_Image_GAN import G  # Your generator model
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import textwrap

# Import evaluation metrics
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
import clip  # OpenAI's CLIP for text-image similarity

# Set the number of images to evaluate (change this value as needed)
num_images = 100

# Load the trained generator model
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
netG = G().to(device)
netG.load_state_dict(torch.load('./saved_models/generator_epoch_199.pth', map_location=device))
netG.eval()  # Set model to evaluation mode

# Load CLIP model
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Define transforms for evaluation (for IS & FID, images are converted to uint8)
image_transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Required for Inception Score & FID
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255).to(torch.uint8))  # Convert to uint8 for FID & IS
])

# Initialize IS and FID metrics (running on CPU)
inception_metric = InceptionScore().to("cpu")
fid_metric = FrechetInceptionDistance().to("cpu")


def load_embedding(dataset_path, split='test', index=0):
    """
    Load a precomputed embedding from the CUB dataset.
    """
    with h5py.File(dataset_path, 'r') as f:
        split_data = f[split]
        example_name = list(split_data.keys())[index]
        example = split_data[example_name]

        # Load the embedding
        embedding = np.array(example['embeddings'], dtype=np.float32)
        embedding = torch.FloatTensor(embedding).unsqueeze(0).to(device)  # Add batch dimension

        # Load the text description
        try:
            text = np.array(example['txt']).astype(str)
        except:
            text = np.array([example['txt'][()].decode('utf-8', errors='replace')])
            text = np.char.replace(text, '�', ' ').astype(str)

        return embedding, text


def generate_image(embedding):
    """
    Generate an image from a given embedding.
    """
    noise = torch.randn(1, 100, 1, 1, device=device)

    with torch.no_grad():
        fake_image = netG(noise, embedding)

    # Convert tensor to image
    fake_image = fake_image.squeeze(0).cpu().numpy()
    fake_image = np.transpose(fake_image, (1, 2, 0))
    fake_image = (fake_image + 1) / 2  # Rescale to [0, 1]
    fake_image = (fake_image * 255).astype(np.uint8)

    return Image.fromarray(fake_image)


def compute_clip_similarity(images, texts):
    """
    Compute the CLIP similarity score between images and their respective text descriptions.
    """
    images_input = torch.stack([clip_preprocess(image) for image in images]).to(device)
    text_inputs = clip.tokenize(texts).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(images_input)
        text_features = clip_model.encode_text(text_inputs)

    # Normalize feature vectors
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features * text_features).sum(dim=-1)  # Cosine similarity
    return similarity.mean().item()


def batch_eval(dataset_path, split='test', num_samples=num_images):
    """
    Evaluate the model on a batch of test samples, computing IS, FID, and CLIP similarity.
    Also displays the generated images in a grid with text annotations.
    """
    generated_images = []
    real_images = []
    texts = []

    # Load test dataset and randomly select indices
    with h5py.File(dataset_path, 'r') as f:
        split_data = f[split]
        all_indices = list(split_data.keys())
        indices = np.random.choice(len(all_indices), size=num_samples, replace=False)

    for idx in indices:
        embedding, text = load_embedding(dataset_path, split, idx)
        generated_image = generate_image(embedding)

        # Apply transforms for evaluation (needed for IS & FID)
        generated_tensor = image_transform(generated_image).cpu()

        # Store images and texts
        generated_images.append(generated_image)
        real_images.append(generated_tensor)  # Here, we assume the evaluation uses generated images for metrics

        # Safely extract text description
        if isinstance(text, np.ndarray):
            if text.size > 0:
                texts.append(str(text.item()))
            else:
                texts.append("Unknown Description")
        elif isinstance(text, str):
            texts.append(text)
        else:
            texts.append("Unknown Description")

    # Compute Inception Score (IS)
    real_images_tensor = torch.stack(real_images)
    is_score = inception_metric(real_images_tensor)
    is_mean, is_std = is_score  # Extract mean and standard deviation

    # Compute FID (here we compare generated images to themselves as a placeholder)
    fid_metric.update(real_images_tensor, real=True)
    fid_metric.update(real_images_tensor, real=False)  # In practice, use real images vs. generated images
    fid_score = fid_metric.compute()

    # Compute CLIP Similarity Score
    clip_score = compute_clip_similarity(generated_images, texts)

    print(f"Inception Score (IS): {is_mean:.4f} ± {is_std:.4f}")
    print(f"Fréchet Inception Distance (FID): {fid_score:.4f}")
    print(f"CLIP Similarity Score: {clip_score:.4f}")


    # Determine grid dimensions
    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))


    # Plot the grid with annotations
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_samples > 1 else [axes]
    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(generated_images[i])
            # Wrap text to avoid overly long titles
            caption = textwrap.fill(texts[i], width=20)
            ax.set_title(caption, fontsize=8)
        ax.axis("off")

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

    '''
    for i in range(rows * cols):
        ax = axes[i]
        if i < num_samples:
            ax.imshow(np.array(generated_images[i]))
            # Wrap text to avoid long titles
            caption = textwrap.fill(texts[i], width=20)
            ax.set_title(caption, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    '''


# Example usage
dataset_path = 'data/birds.hdf5'
batch_eval(dataset_path, split='test', num_samples=num_images)
