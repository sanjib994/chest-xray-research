"""
VinBigData_XAI — Production Preprocessing
Matches the notebook training/inference experiment:
- grayscale to RGB
- resize to 224x224
- ImageNet normalization
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_inference_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_image(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as image:
        return image.convert("L").copy()

def preprocess_image(path):
    image = load_image(path)
    transform = build_inference_transform()
    return transform(image)

def preprocess_batch(paths):
    tensors = [preprocess_image(path) for path in paths]
    if not tensors:
        raise ValueError("No image paths supplied.")
    return torch.stack(tensors)

def tensor_to_numpy(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Expected torch.Tensor.")
    return tensor.detach().cpu().numpy()