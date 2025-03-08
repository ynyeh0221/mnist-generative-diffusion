# Efficient Transformer-Based Diffusion Model for MNIST

This project implements a lightweight Transformer-based diffusion model for generating MNIST handwritten digit images from noise. The model incorporates several optimizations to improve computational efficiency while maintaining generation quality.

## Features

- **Patch-Based Approach**: Uses image patches similar to Vision Transformers (ViT) instead of individual pixels
- **Efficient Linear Attention**: Implements O(n) complexity attention mechanism instead of standard O(n²) self-attention
- **Conditional Generation**: Generates specific digits (0-9) based on class labels
- **Multi-Step Diffusion Process**: Employs a progressive denoising approach from pure noise to clean images
- **Optimized Architecture**: Reduced complexity with fewer layers and optimized hyperparameters

## Requirements

- PyTorch
- torchvision
- matplotlib
- GPU (optional, but recommended for faster training)

## Installation

```bash
pip install torch torchvision matplotlib
```

## Usage

Simply run the script to train the model and generate MNIST digit images:

```bash
python mnist_diffusion.py
```

The script will:
1. Download the MNIST dataset
2. Train the diffusion model for 30 epochs
3. Generate sample images for all digits 0-9
4. Display the progressive generation process from noise to clear digits

## Model Architecture

The implementation consists of several key components:

- **PatchEmbedding**: Converts input images into patch embeddings
- **LinearAttention**: Efficient attention mechanism that avoids O(n²) complexity
- **EfficientTransformerBlock**: Optimized transformer block with Pre-Norm structure and simplified residual connections
- **EfficientTransformerDenoiser**: Core model that handles the transformation from noise to images

## Key Hyperparameters

- Image Size: 16×16 pixels
- Patch Size: 2×2 pixels
- Embedding Dimension: 128
- Number of Attention Heads: 4
- Number of Transformer Layers: 3
- Feedforward Dimension: 256
- Batch Size: 128
- Learning Rate: 3e-4 with cosine annealing warm restarts
- Weight Decay: 0.05
- Number of Diffusion Steps: 10

## Performance Optimizations

1. **Adjusted Input Resolution**: Using 16×16 resolution for balance between speed and detail
2. **Decreased Model Complexity**: Smaller transformer blocks with fewer layers
3. **Patch-Based Representation**: Using 2×2 patches instead of individual pixels
4. **Linear Attention Mechanism**: Reducing complexity from O(n²) to O(n)
5. **Batch Size Optimization**: Using batch size of 128 for better training stability
6. **Learning Rate Scheduling**: Cosine annealing with warm restarts for improved convergence

## Training Process

The model is trained using a multi-step diffusion approach:
1. For each training step, a random diffusion step t is selected
2. Noise is added to the original images based on the diffusion step
3. The model is trained to predict the original clean images
4. Mean Squared Error (MSE) loss is used to optimize the model parameters

## Visualization

The script generates a visualization showing the progressive generation process from random noise to clear digits. For each digit 0-9:
- Starting with pure random noise
- Going through 10 denoising steps
- Ending with a clean digit image

## Device Support

The script automatically detects and uses:
- CUDA (NVIDIA GPU)
- MPS (Apple Silicon)
- CPU (fallback)

## Potential Extensions

- Increase image resolution
- Apply to more complex datasets
- Implement unconditional generation
- Add more advanced sampling techniques
