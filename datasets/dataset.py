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

class Oxford102Dataset(Dataset):
    def __init__(self, base_dir, split='train', embeddings_path=None, transform=None):
        """
        Args:
            base_dir (str): Path to dataset root (contains train/valid/test folders)
            split (str): One of 'train', 'valid', or 'test'
            embeddings_path (str): Path to .npz file with precomputed embeddings
                                  (keys in format "images/image_XXXXX.jpg")
            transform (callable, optional): Optional transforms for images
        """
        self.base_dir = base_dir
        self.split = split
        self.transform = transform or self.default_transform()
        self.embeddings = self.load_embeddings(embeddings_path) if embeddings_path else None

        # Collect all image paths and class labels
        self.image_paths = []
        self.class_labels = []

        split_dir = os.path.join(base_dir, split)

        # Iterate through class folders (1, 2, 3,...)
        for class_folder in sorted(os.listdir(split_dir)):
            if not class_folder.isdigit():
                continue

            class_idx = int(class_folder) - 1  # Convert to 0-based index
            class_dir = os.path.join(split_dir, class_folder)

            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(split_dir, class_folder, img_file)
                    self.image_paths.append(img_path)
                    self.class_labels.append(class_idx)

        # Verify embeddings if provided
        if self.embeddings:
            self.verify_embeddings()

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_embeddings(self, path):
        """Load embeddings with error handling"""
        try:
            embeddings = np.load(path, allow_pickle=True)
            # Convert to dict if it's an npz file
            return dict(embeddings) if isinstance(embeddings, np.lib.npyio.NpzFile) else embeddings
        except Exception as e:
            raise ValueError(f"Failed to load embeddings from {path}: {str(e)}")

    def verify_embeddings(self):
        """Verify all images have corresponding embeddings in format 'images/image_XXXXX.jpg'"""
        missing = []
        matched = 0

        for img_path in self.image_paths:
            # Generate the expected embedding key
            img_name = os.path.basename(img_path)
            embedding_key = f"images/{img_name}"

            if embedding_key not in self.embeddings:
                missing.append(img_path)
            else:
                matched += 1

        if missing:
            print(f"\nEmbedding Status: {matched}/{len(self.image_paths)} images have embeddings")
            print("First 5 missing embeddings:")
            for path in missing[:5]:
                img_name = os.path.basename(path)
                print(f"  - Expected key: 'images/{img_name}'")

            print("\nSample actual embedding keys:")
            sample_keys = list(self.embeddings.keys())
            print(f"  - {sample_keys[0]}") if sample_keys else print("  - No keys found")
            print(f"  - {sample_keys[1]}") if len(sample_keys) > 1 else ""

            # Uncomment to raise error if strict matching is required
            # raise ValueError(f"{len(missing)} images missing embeddings")

    def get_embedding_key(self, img_path):
        """Convert image path to embedding key format 'images/image_XXXXX.jpg'"""
        return f"images/{os.path.basename(img_path)}"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Get class label
        class_label = torch.LongTensor([self.class_labels[idx]])

        # Initialize sample
        sample = {
            "image": image,                   # [3, 128, 128]
            "class_label": class_label,       # [1]
            "image_path": img_path           # For debugging
        }

        # Add embeddings if available
        if self.embeddings:
            embedding_key = self.get_embedding_key(img_path)
            if embedding_key in self.embeddings:
                sample["text_embeddings"] = torch.FloatTensor(self.embeddings[embedding_key])  # [10, 384]
            else:
                # Create zero embeddings if missing (optionally log warning)
                sample["text_embeddings"] = torch.zeros(10, 384)
                if idx < 3:  # Only show first few warnings
                    print(f"Warning: Using zero embeddings for {embedding_key}")

        return sample
