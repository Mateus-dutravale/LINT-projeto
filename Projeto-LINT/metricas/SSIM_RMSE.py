import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def _mse(imageA, imageB):
    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    return err

def _check_and_resize(image1, image2):
    """Verifica e ajusta as dimensões das imagens, se necessário."""

    # Reduz para grayscale se a imagem tiver 3 canais
    if image1.ndim == 3:
        image1 = image1[:, :, 0]
    if image2.ndim == 3:
        image2 = image2[:, :, 0]

    ho, wo = image1.shape
    hc, wc = image2.shape

    ratio_orig = ho / wo
    ratio_comp = hc / wc

    if round(ratio_orig, 2) != round(ratio_comp, 2):
        raise ValueError("As imagens não têm a mesma proporção.")

    if ho > hc and wo > wc:
        image1 = Image.resize(image1, (hc, wc), anti_aliasing=True)
    elif ho < hc and wo < wc:
        raise ValueError("A imagem aprimorada é maior do que a original.")

    return image1, image2

def calcular_rmse(img1, img2):
    if isinstance(img1, np.ndarray):
        gray1 = img1.astype(np.float64)
    else:
        gray1 = np.array(img1.convert("L")).astype(np.float64)

    if isinstance(img2, np.ndarray):
        gray2 = img2.astype(np.float64)
    else:
        gray2 = np.array(img2.convert("L")).astype(np.float64)

    img1, img2 = _check_and_resize(img1, img2)
    return np.sqrt(_mse(img1, img2))

def calcular_ssim(img1, img2):
    # Converta para float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Normaliza para [0, 1] se estiver em 8 bits
    if img1.max() > 1.0:
        img1 /= 255.0
    if img2.max() > 1.0:
        img2 /= 255.0

    # Constantes para estabilidade
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Filtros gaussiano
    mu1 = gaussian_filter(img1, sigma=1.5)
    mu2 = gaussian_filter(img2, sigma=1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 * img1, sigma=1.5) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sigma=1.5) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=1.5) - mu1_mu2

    # Fórmula do SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(ssim_map)
