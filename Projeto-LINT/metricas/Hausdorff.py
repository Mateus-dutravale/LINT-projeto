import torch
from PIL import Image
import torchvision.transforms as transforms
import os

def load_image_as_BLA(image_path):
    """Carrega a imagem em tons de cinza no formato (B, L, A) → (1, width, height)."""
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    tensor = transforms.ToTensor()(image)        # (1, H, W)
    tensor = tensor.permute(0, 2, 1)             # (1, W, H) = (B, L, A)
    return tensor

def torch2D_Hausdorff_distance(x, y):
    """Calcula a distância de Hausdorff entre duas imagens no formato (B, L, A)."""
    x = x.float()
    y = y.float()

    distance_matrix = torch.cdist(x, y, p=2)  # (B, L, A) entre si
    value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]

    value = torch.cat((value1, value2), dim=1)
    return value.max(1)[0]

def calcular_hausdorff(imagem_original_path, imagem_aprimorada_path):
    """
    Interface principal da métrica. Carrega as imagens e retorna a distância de Hausdorff.
    """
    img1 = load_image_as_BLA(imagem_original_path)
    img2 = load_image_as_BLA(imagem_aprimorada_path)
    return torch2D_Hausdorff_distance(img1, img2).item()

# Teste manual
if __name__ == "__main__":
    caminho_u = "teste1.jpg"
    caminho_v = "teste2.jpg"

    if not os.path.exists(caminho_u) or not os.path.exists(caminho_v):
        print("Imagens não encontradas.")
    else:
        resultado = calcular_hausdorff(caminho_u, caminho_v)
        print("Distância de Hausdorff:", resultado)
