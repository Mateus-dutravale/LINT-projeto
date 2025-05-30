import os
import cv2
from metricas.SSIM_RMSE import calcular_rmse, calcular_ssim
from metricas.SNR import calcular_snr
from metricas.Hausdorff import calcular_hausdorff
from metricas.PSNR import calcular_psnr

# ============== CONFIGURAÇÕES ============== (Altere aqui!)
MODO = 'pasta'  # 'pasta' ou 'arquivo'

# Configurações para modo PASTA
PASTA_ORIGINAIS = "Projeto-LINT/script/downgrade/blurred"
PASTA_APRIMORADAS = "Projeto-LINT/script/enhancement/scunet/blurred"

# Configurações para modo ARQUIVO
CAMINHO_ORIGINAL = "Projeto-LINT/script/downgrade/blurred/2nd_camera_online_for_the_Long_Incident_Fire_seen_from_Bald_Mt_at_842_AM_FR-1240_blur.jpg"
CAMINHO_APRIMORADA = "Projeto-LINT/script/enhancement/scunet/blurred/2nd_camera_online_for_the_Long_Incident_Fire_seen_from_Bald_Mt_at_842_AM_FR-1240_blur.jpg"

# Métricas a serem calculadas (True/False)
CALCULAR_SSIM_RMSE = False
CALCULAR_SNR = True
CALCULAR_HAUSDORFF = True
CALCULAR_PSNR = True
# ===========================================

def carregar_imagens(pasta_originais, pasta_aprimoradas):
    arquivos_originais = [f for f in sorted(os.listdir(pasta_originais)) 
                         if f.lower().endswith(('.jpg', '.jpeg'))]
    arquivos_aprimoradas = [f for f in sorted(os.listdir(pasta_aprimoradas)) 
                           if f.lower().endswith(('.jpg', '.jpeg'))]
    
    return [
        (os.path.join(pasta_originais, nome), os.path.join(pasta_aprimoradas, nome))
        for nome in arquivos_originais if nome in arquivos_aprimoradas
    ]

def carregar_imagem_unica(original, aprimorada):
    if not all([original.lower().endswith(('.jpg', '.jpeg')), 
                aprimorada.lower().endswith(('.jpg', '.jpeg'))]):
        raise ValueError("Apenas arquivos .jpg ou .jpeg são suportados.")
    return [(original, aprimorada)]

def aplicar_metricas(pares):
    resultados = []
    for img1_path, img2_path in pares:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"Erro ao carregar: {img1_path} ou {img2_path}")
            continue

        resultado = {'nome': os.path.basename(img1_path)}
        
        if CALCULAR_SSIM_RMSE:
            resultado['SSIM'] = round(calcular_ssim(img1, img2), 4)
            resultado['RMSE'] = round(calcular_rmse(img1, img2), 4)
        
        if CALCULAR_SNR:
            resultado['SNR'] = round(calcular_snr(img1, img2), 4)
        
        if CALCULAR_HAUSDORFF:
            resultado['Hausdorff'] = round(calcular_hausdorff(img1_path, img2_path), 4)
        
        if CALCULAR_PSNR:
            resultado['PSNR'] = round(calcular_psnr(img1, img2), 4)
        
        resultados.append(resultado)
    return resultados

def imprimir_resultados(resultados):
    if not resultados:
        print("Nenhum resultado válido.")
        return

    # Filtra colunas ativas
    colunas = ['Imagem']
    if CALCULAR_SSIM_RMSE: colunas.extend(['SSIM', 'RMSE'])
    if CALCULAR_SNR: colunas.append('SNR')
    if CALCULAR_HAUSDORFF: colunas.append('Hausdorff')
    if CALCULAR_PSNR: colunas.append('PSNR')

    print("\n=== RESULTADOS ===")
    print(" | ".join(colunas))
    print("-" * 50)
    
    for r in resultados:
        print(" | ".join([r['nome']] + [str(r.get(c, '-')) for c in colunas[1:]]))

    # Médias (apenas para modo pasta)
    if MODO == 'pasta' and len(resultados) > 1:
        print("\n=== MEDIAS ===")
        for metrica in colunas[1:]:
            media = sum(float(r[metrica]) for r in resultados) / len(resultados)
            print(f"{metrica}: {round(media, 4)}")

if __name__ == "__main__":
    try:
        # Seleção do modo
        if MODO == 'pasta':
            pares = carregar_imagens(PASTA_ORIGINAIS, PASTA_APRIMORADAS)
        elif MODO == 'arquivo':
            pares = carregar_imagem_unica(CAMINHO_ORIGINAL, CAMINHO_APRIMORADA)
        else:
            raise ValueError("Modo deve ser 'pasta' ou 'arquivo'")

        resultados = aplicar_metricas(pares)
        imprimir_resultados(resultados)

    except Exception as e:
        print(f"\nERRO: {str(e)}")