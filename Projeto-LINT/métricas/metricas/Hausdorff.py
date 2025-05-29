import torch
from PIL import Image
import torchvision.transforms as transforms

def _load_image_as_tensor(image_path):
    """Carrega a imagem e a transforma em tensor normalizado entre [0,1]."""
    image = Image.open(image_path).convert("RGB")  # Garante 3 canais
    transformation = transforms.Compose([
        transforms.ToTensor()
    ])
    return transformation(image)

def calcular_hausdorff(img1, img2):
    """
    Calcula a distância de Hausdorff aproximada entre duas imagens (como tensores PyTorch).
    As imagens devem estar normalizadas em [0,1] e no formato (C, H, W).
    """
    if isinstance(img1, str):
        img1 = _load_image_as_tensor(img1)
    if isinstance(img2, str):
        img2 = _load_image_as_tensor(img2)

    if img1.shape != img2.shape:
        raise ValueError("As imagens devem ter o mesmo shape para Hausdorff.")

    # Flatten as imagens para (H*W, C)
    B, H, W = 1, img1.shape[1], img1.shape[2]
    img1_flat = img1.view(1, -1, 3)  # (1, H*W, C)
    img2_flat = img2.view(1, -1, 3)

    distance_matrix = torch.cdist(img1_flat, img2_flat, p=2)  # matriz de distâncias euclidianas

    forward_hd = distance_matrix.min(2)[0].max(1)[0]
    backward_hd = distance_matrix.min(1)[0].max(1)[0]

    hausdorff_distance = torch.max(forward_hd, backward_hd)
    return hausdorff_distance.item()

