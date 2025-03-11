# Conditional CNN Denoiser for MNIST Digits

## Overview
This project implements a **Conditional Convolutional Neural Network (CNN) Denoiser** using **PyTorch**. The model is designed to reconstruct MNIST digits from noisy inputs, incorporating conditional label embeddings to improve performance. A multi-step diffusion approach is employed for progressive denoising.

## Features
- **Conditional Denoising:** The model uses **label embeddings** to condition the denoising process on digit class information.
- **Multi-Step Diffusion:** The denoiser progressively refines images through multiple steps.
- **CNN-based Architecture:** Convolutional and transposed convolutional layers for encoding and decoding noisy inputs.
- **Learning Rate Scheduling:** A step-based scheduler adjusts the learning rate over epochs.

## Dependencies
Make sure you have the following Python packages installed:
```bash
pip install torch torchvision matplotlib
```

## Dataset
The model is trained on the **MNIST dataset**, which consists of 28x28 grayscale handwritten digit images. The dataset is automatically downloaded using `torchvision.datasets.MNIST`.

## Model Architecture
The `ConditionalDenoiser` model consists of:
- **Label Embedding Layer**: Converts class labels into a 64-dimensional embedding.
- **Encoder**: A series of convolutional layers with BatchNorm and ReLU activation.
- **Decoder**: Transposed convolutional layers to reconstruct the original image.

## Training Procedure
1. Load and preprocess the MNIST dataset (normalize and convert to tensors).
2. Add noise to input images using a randomly selected **diffusion step**.
3. Train the model using **Mean Squared Error (MSE) loss** and **Adam optimizer**.
4. Apply a **StepLR scheduler** to adjust the learning rate dynamically.
5. Run for `30` epochs, evaluating the model at each step.

## Training Script
```python
# Initialize the model
model = ConditionalDenoiser().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.MSELoss()

# Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        t = random.randint(1, num_diffusion_steps)
        noise_level = torch.tensor([t / num_diffusion_steps], device=device).view(-1, 1, 1, 1)
        noisy_images = images + torch.randn_like(images).to(device) * noise_level
        
        optimizer.zero_grad()
        outputs = model(noisy_images, noise_level, labels)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')
```

## Evaluation & Visualization
After training, the model can generate denoised images progressively using a **multi-step diffusion approach**. A sample noisy image is denoised iteratively, and each step is visualized using `matplotlib`.

```python
model.eval()
prompt = random.randint(0, 9)
with torch.no_grad():
    denoised_img = torch.randn((1, 1, 28, 28)).to(device)
    label = torch.tensor([prompt], device=device)
    
    fig, axes = plt.subplots(1, num_diffusion_steps + 1, figsize=(15, 3))
    for t in reversed(range(1, num_diffusion_steps + 1)):
        noise_level = torch.tensor([[[[t / num_diffusion_steps]]]], device=device)
        denoised_img = model(denoised_img, noise_level, label)
        axes[num_diffusion_steps - t + 1].imshow(denoised_img.cpu().detach().squeeze(), cmap='gray')
        axes[num_diffusion_steps - t + 1].axis('off')
plt.show()
```

## Results
The denoising process progressively restores the noisy images back to clear MNIST digits, demonstrating the effectiveness of **conditional denoising** and **multi-step refinement**.

## Usage
1. **Train the model** using the provided script.
2. **Evaluate and visualize** the denoising process.
3. **Modify and experiment** with different noise levels, diffusion steps, and model architectures.
