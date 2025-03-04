import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Improved Conditional CNN Denoiser Model (Encoder-Decoder with label embedding)
class ConditionalDenoiser(nn.Module):
    def __init__(self, num_classes=10):
        super(ConditionalDenoiser, self).__init__()
        # Increased embedding size to improve label representation
        self.label_embedding = nn.Embedding(num_classes, 64)
        self.label_fc = nn.Linear(64, 28*28)
        # Added extra convolutional layers to improve feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1), nn.Tanh()
        )

    def forward(self, x, noise_level, labels):
        label_embedding = self.label_fc(self.label_embedding(labels)).view(-1, 1, 28, 28)
        noise_channel = noise_level.expand_as(x)
        input_combined = torch.cat([x, noise_channel, label_embedding], dim=1)
        encoded = self.encoder(input_combined)
        decoded = self.decoder(encoded)
        return decoded

# Initialize model, optimizer, loss function, and scheduler
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Current using device：{device}")
model = ConditionalDenoiser().to(device)
# Lowered learning rate to stabilize training
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Added scheduler to gradually reduce learning rate for better convergence
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.MSELoss()

# Training loop
epochs = 30  # Increased epochs for improved performance
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # Reduced noise level complexity to simplify training
        noise_level = torch.rand(images.size(0), 1, 1, 1).to(device) * 0.3
        noisy_images = images + torch.randn_like(images).to(device) * noise_level

        optimizer.zero_grad()
        outputs = model(noisy_images, noise_level, labels)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

# Evaluate and visualize generation from random prompt
model.eval()
prompt = random.randint(0, 9)
with torch.no_grad():
    noisy_img = torch.randn((1, 1, 28, 28)).to(device)
    denoised_img = noisy_img.clone()
    label = torch.tensor([prompt], device=device)

    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    axes[0].imshow(noisy_img.cpu().squeeze(), cmap='gray')
    axes[0].set_title(f"Start (Noise)")
    axes[0].axis('off')

    for i in range(1, 6):
        noise_level = torch.tensor([[[[0.5 - (i * 0.1)]]]], device=device)
        denoised_img = model(denoised_img, noise_level, label)
        axes[i].imshow(denoised_img.cpu().detach().squeeze(), cmap='gray')
        axes[i].set_title(f"Step {i}\nDigit {prompt}")
        axes[i].axis('off')

plt.tight_layout()
plt.show()
