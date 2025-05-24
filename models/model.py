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



class TextEmbeddingNet(nn.Module):
    """Process Sentence-BERT embeddings (10 captions → 1 aggregated vector)"""
    def __init__(self, text_embed_dim=384, text_latent_dim=128):
        super(TextEmbeddingNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_embed_dim, text_latent_dim),
            nn.BatchNorm1d(text_latent_dim),
            nn.LeakyReLU(0.2)
        )
        # Optional: Use LSTM/Attention for caption aggregation (uncomment if needed)
        # self.lstm = nn.LSTM(text_embed_dim, text_latent_dim, batch_first=True)

    def forward(self, text_embeddings):
        # Input: [batch_size, 10, 384] (10 captions per image)

        # Option 1: Mean pooling (simple)
        text_embedding = text_embeddings.mean(dim=1)  # → [batch_size, 384]

        # Option 2: LSTM (advanced)
        # _, (hidden, _) = self.lstm(text_embeddings)
        # text_embedding = hidden[-1]  # → [batch_size, text_latent_dim]

        return self.fc(text_embedding)  # → [batch_size, text_latent_dim]

class Generator(nn.Module):
    """Generator with Sentence-BERT support"""
    def __init__(self, noise_dim=100, text_latent_dim=128, ngf=64):
        super(Generator, self).__init__()
        self.text_net = AttentionProjection(text_latent_dim=text_latent_dim)

        # Project noise + text latent to conv space
        self.fc = nn.Linear(noise_dim + text_latent_dim, 8 * 8 * 8 * ngf)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(8 * ngf, 4 * ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # 128x128x3
        )

    def forward(self, noise, text_embeddings):
        # Process text: [batch_size, 10, 384] → [batch_size, text_latent_dim]
        text_latent = self.text_net(text_embeddings)
        # Concatenate noise + text latent
        x = torch.cat([noise, text_latent], dim=1)
        x = self.fc(x).view(-1, 8 * 64, 8, 8)
        return self.main(x)

class Discriminator(nn.Module):
    """Discriminator with Sentence-BERT support"""
    def __init__(self, text_latent_dim=128, ndf=64, num_classes=102):
        super(Discriminator, self).__init__()
        self.text_net = TextEmbeddingNet(text_embed_dim=384, text_latent_dim=text_latent_dim)

        # Image processing
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, 2 * ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * ndf, 4 * ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * ndf, 8 * ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8 * ndf),
            nn.LeakyReLU(0.2, inplace=True),  # 8x8 x (8*ndf)
        )

        # Text-conditioned heads
        self.disc_head = nn.Sequential(
            nn.Conv2d(8 * ndf + text_latent_dim, 1, 8, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(8 * ndf + text_latent_dim, num_classes, 8, 1, 0, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, image, text_embeddings):
        # Process image
        img_features = self.main(image)  # [batch_size, 8*ndf, 8, 8]
        # Process text: [batch_size, 10, 384] → [batch_size, text_latent_dim]
        text_latent = self.text_net(text_embeddings)
        text_latent = text_latent.view(-1, text_latent_dim, 1, 1).repeat(1, 1, 8, 8)
        # Concatenate
        x = torch.cat([img_features, text_latent], dim=1)
        # Predictions
        real_fake_prob = self.disc_head(x).view(-1)
        class_prob = self.aux_head(x).view(-1, num_classes)
        return real_fake_prob, class_prob