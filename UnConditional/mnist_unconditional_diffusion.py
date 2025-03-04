import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Improved CNN Denoiser Model (Encoder-Decoder)
class ImprovedDenoiser(nn.Module):
    def __init__(self):
        super(ImprovedDenoiser, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1), nn.Tanh()
        )

    def forward(self, x, noise_level):
        noise_channel = noise_level.expand_as(x)
        input_combined = torch.cat([x, noise_channel], dim=1)
        encoded = self.encoder(input_combined)
        decoded = self.decoder(encoded)
        return decoded

# Initialize model, optimizer, and loss function
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Current using device：{device}")
model = ImprovedDenoiser().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for images, _ in train_loader:
        images = images.to(device)
        noise_level = torch.rand(images.size(0), 1, 1, 1).to(device)
        noisy_images = images + torch.randn_like(images).to(device) * noise_level

        optimizer.zero_grad()
        outputs = model(noisy_images, noise_level)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

# Evaluate and visualize generation from random noise
model.eval()
with torch.no_grad():
    noisy_img = torch.randn((1, 1, 28, 28)).to(device)
    denoised_img = noisy_img.clone()

    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    axes[0].imshow(noisy_img.cpu().squeeze(), cmap='gray')
    axes[0].set_title("Start (Noise)")
    axes[0].axis('off')

    for i in range(1, 6):
        noise_level = torch.tensor([[[[0.5 - (i * 0.1)]]]], device=device)
        denoised_img = model(denoised_img, noise_level)
        axes[i].imshow(denoised_img.cpu().detach().squeeze(), cmap='gray')
        axes[i].set_title(f"Step {i}")
        axes[i].axis('off')

plt.tight_layout()
plt.show()
