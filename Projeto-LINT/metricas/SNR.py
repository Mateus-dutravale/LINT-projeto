import cv2
import numpy as np

def calcular_snr(image1, image2):
    """Calcula a razão sinal-ruído (SNR) entre duas imagens BGR."""

    # Converter para escala de cinza se necessário
    if len(image1.shape) == 3:
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        image1_gray = image1.copy()

    if len(image2.shape) == 3:
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        image2_gray = image2.copy()

    # Verificar proporção e redimensionar se necessário
    ho, wo = image1_gray.shape
    hc, wc = image2_gray.shape
    ratio_orig = ho / wo
    ratio_comp = hc / wc

    if round(ratio_orig, 2) != round(ratio_comp, 2):
        raise ValueError("As imagens não têm a mesma proporção.")

    if ho > hc and wo > wc:
        image1_gray = cv2.resize(image1_gray, (wc, hc))
    elif ho < hc and wo < wc:
        raise ValueError("A imagem aprimorada é maior que a original.")

    # Cálculo da SNR
    signal_power = np.sum(image1_gray.astype(np.float64) ** 2)
    noise = image1_gray.astype(np.float64) - image2_gray.astype(np.float64)
    noise_power = np.sum(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return snr
