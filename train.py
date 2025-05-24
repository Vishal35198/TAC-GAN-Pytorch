import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np


# Initialize models
G = Generator(noise_dim=noise_dim, text_latent_dim=text_latent_dim).to(device)
D = Discriminator(text_latent_dim=text_latent_dim, num_classes=num_classes).to(device)

# Optimizers
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss functions
criterion_adv = nn.BCELoss()  # Real/Fake loss
criterion_aux = nn.CrossEntropyLoss()  # Class loss

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Fixed noise + text for visualization
fixed_noise = torch.randn(8, noise_dim, device=device)
fixed_text = torch.randn(8, 10, text_embed_dim, device=device)


def train_tacgan(
    G, D,
    train_loader,
    optimizer_G, optimizer_D,
    criterion_adv, criterion_aux,
    device, epochs,
    fixed_noise, fixed_text,
    checkpoint_dir="checkpoints",
    image_dir="generated_images"
):
    """
    Train TAC-GAN model

    Args:
        G: Generator
        D: Discriminator
        train_loader: DataLoader for training set
        optimizers: Optimizers for G and D
        criteria: Loss functions (adversarial and auxiliary)
        device: torch.device
        epochs: Number of training epochs
        fixed_noise: Fixed noise for visualization
        fixed_text: Fixed text embeddings for visualization
    """
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # Loss tracking
    history = {
        'D_loss': [],
        'G_loss': [],
        'D_real_acc': [],
        'D_fake_acc': []
    }

    for epoch in range(epochs):
        D_losses = []
        G_losses = []
        real_acc = []
        fake_acc = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            # Prepare batch data
            real_images = batch["image"].to(device)
            text_embeddings = batch["text_embeddings"].to(device)
            class_labels = batch["class_label"].squeeze().to(device)
            batch_size = real_images.size(0)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            D.zero_grad()

            # Real images
            real_labels = torch.ones(batch_size, device=device)
            real_fake_prob_real, class_prob_real = D(real_images, text_embeddings)
            d_loss_real = criterion_adv(real_fake_prob_real, real_labels)
            d_loss_class = criterion_aux(class_prob_real, class_labels)

            # Calculate accuracy for real images
            real_pred = real_fake_prob_real > 0.5
            real_acc.append(real_pred.float().mean().item())

            # Fake images
            noise = torch.randn(batch_size, noise_dim, device=device)
            with torch.no_grad():
                fake_images = G(noise, text_embeddings)

            fake_labels = torch.zeros(batch_size, device=device)
            real_fake_prob_fake, _ = D(fake_images.detach(), text_embeddings)
            d_loss_fake = criterion_adv(real_fake_prob_fake, fake_labels)

            # Calculate accuracy for fake images
            fake_pred = real_fake_prob_fake < 0.5
            fake_acc.append(fake_pred.float().mean().item())

            # Wrong images (random image-text mismatch)
            wrong_idx = torch.randperm(batch_size)
            # added wrong capt_idx
            wrong_capt_idx = torch.randperm(batch_size)
            real_fake_prob_wrong, class_prob_wrong = D(
                real_images[wrong_idx],
                text_embeddings[wrong_capt_idx]
            )
            d_loss_wrong = criterion_adv(real_fake_prob_wrong, fake_labels)
            d_loss_wrong_class = criterion_aux(class_prob_wrong, class_labels[wrong_idx])

            # Total D loss
            d_loss = d_loss_real + d_loss_fake + d_loss_wrong + d_loss_class + d_loss_wrong_class
            d_loss.backward()
            optimizer_D.step()
            D_losses.append(d_loss.item())

            # -----------------
            #  Train Generator
            # -----------------
            G.zero_grad()

            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_images = G(noise, text_embeddings)
            real_fake_prob_fake, class_prob_fake = D(fake_images, text_embeddings)

            # Generator wants D to think fake images are real
            g_loss_adv = criterion_adv(real_fake_prob_fake, real_labels)
            g_loss_class = criterion_aux(class_prob_fake, class_labels)
            g_loss = g_loss_adv + g_loss_class
            g_loss.backward()
            optimizer_G.step()
            G_losses.append(g_loss.item())

            # Update progress bar
            progress_bar.set_postfix({
                "D Loss": d_loss.item(),
                "G Loss": g_loss.item(),
                "D Acc Real": real_pred.float().mean().item(),
                "D Acc Fake": fake_pred.float().mean().item()
            })

        # Save epoch statistics
        history['D_loss'].append(torch.mean(torch.tensor(D_losses)).item())
        history['G_loss'].append(torch.mean(torch.tensor(G_losses)).item())
        history['D_real_acc'].append(torch.mean(torch.tensor(real_acc)).item())
        history['D_fake_acc'].append(torch.mean(torch.tensor(fake_acc)).item())

        # Save generated images
        with torch.no_grad():
            G.eval()
            fake_images = G(fixed_noise, fixed_text)
            save_image(
                fake_images,
                os.path.join(image_dir, f"epoch_{epoch+1}.png"),
                nrow=4,
                normalize= False
            )
            G.train()

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch + 1,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'D_loss': history['D_loss'][-1],
                'G_loss': history['G_loss'][-1],
            }, os.path.join(checkpoint_dir, f"tacgan_epoch_{epoch+1}.pth"))

        print(f"\nEpoch {epoch+1}/{epochs} | "
              f"D Loss: {history['D_loss'][-1]:.4f} | "
              f"G Loss: {history['G_loss'][-1]:.4f} | "
              f"D Acc Real: {history['D_real_acc'][-1]:.2f} | "
              f"D Acc Fake: {history['D_fake_acc'][-1]:.2f}")

    return history

history = train_tacgan(G=G,
             D=D,
             train_loader=dataloader,
             optimizer_G=optimizer_G,
             optimizer_D=optimizer_D,
             criterion_adv=criterion_adv,
             criterion_aux=criterion_aux,
             device = device,
             epochs=10,
             fixed_noise=fixed_noise,
             fixed_text=fixed_text
)