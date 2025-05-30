import cv2
import numpy as np
#from skimage import structural_similarity as ssim 

def _mse(imageA, imageB):
    """Calcula o erro quadrático médio entre duas imagens."""
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def _check_and_resize(image1, image2):
    """Verifica e ajusta as dimensões das imagens, se necessário."""
    ho, wo = image1.shape
    hc, wc = image2.shape

    ratio_orig = ho / wo
    ratio_comp = hc / wc

    if round(ratio_orig, 2) != round(ratio_comp, 2):
        raise ValueError("As imagens não têm a mesma proporção.")

    if ho > hc and wo > wc:
        # Redimensionar a imagem maior para a menor
        image1 = cv2.resize(image1, (wc, hc))
    elif ho < hc and wo < wc:
        raise ValueError("A imagem aprimorada é maior do que a original.")

    return image1, image2

def calcular_ssim(img1, img2):
    """
    Usa a implementação do SSIM do OpenCV (se disponível).
    Requer OpenCV >= 4.4.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Verifica e redimensiona (mantendo sua função _check_and_resize)
    gray1, gray2 = _check_and_resize(gray1, gray2)
    
    # Calcula SSIM usando OpenCV
    return cv2.quality.QualitySSIM_compute(gray1, gray2)[0]

def calcular_rmse(img1, img2):
    """Calcula o RMSE entre duas imagens BGR."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    gray1, gray2 = _check_and_resize(gray1, gray2)
    return np.sqrt(_mse(gray1, gray2))
