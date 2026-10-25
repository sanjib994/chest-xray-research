"""
VinBigData_XAI — Production Transform Definitions
"""

from torchvision import transforms


IMAGE_SIZE = (224, 224)

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


def inference_transforms():
    """
    Deterministic inference transform.
    """

    return transforms.Compose([
        transforms.Resize(
            IMAGE_SIZE
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def validation_transforms():
    """
    Deterministic validation transform.

    Kept separate from inference so the product layer can evolve
    without changing the public inference interface.
    """

    return inference_transforms()
