import os
import argparse
import cv2
from metricas.SSIM_RMSE import calcular_rmse, calcular_ssim
from metricas.SNR import calcular_snr
from metricas.Hausdorff import calcular_hausdorff
from metricas.PSNR import calcular_psnr

def carregar_imagens(pasta_perfeita, pasta_aprimorada):
    # Listar apenas arquivos JPG e JPEG
    arquivos_perfeita = [f for f in sorted(os.listdir(pasta_perfeita)) 
                        if f.lower().endswith(('.jpg', '.jpeg'))]
    arquivos_aprimorada = [f for f in sorted(os.listdir(pasta_aprimorada)) 
                          if f.lower().endswith(('.jpg', '.jpeg'))]
    
    pares = []
    for nome in arquivos_perfeita:
        if nome in arquivos_aprimorada:
            path1 = os.path.join(pasta_perfeita, nome)
            path2 = os.path.join(pasta_aprimorada, nome)
            pares.append((path1, path2))
    return pares

def aplicar_metricas(pares, usar_ssim_rmse, usar_snr, usar_hausdorff, usar_psnr):
    resultados = []
    for idx, (img1_path, img2_path) in enumerate(pares):
        resultado = {
            "nome": os.path.basename(img1_path)
        }

        imagem1 = cv2.imread(img1_path)
        imagem2 = cv2.imread(img2_path)

        if imagem1 is None or imagem2 is None:
            print(f"Erro ao carregar imagens: {img1_path} ou {img2_path}")
            continue

        if usar_ssim_rmse:
            try:
                ssim_val = calcular_ssim(imagem1, imagem2)
                rmse_val = calcular_rmse(imagem1, imagem2)
                resultado["SSIM"] = round(ssim_val, 4)
                resultado["RMSE"] = round(rmse_val, 4)
            except ValueError as e:
                resultado["SSIM"] = resultado["RMSE"] = f"Erro: {str(e)}"

        if usar_snr:
            snr_val = calcular_snr(imagem1, imagem2)
            resultado["SNR"] = round(snr_val, 4)

        if usar_hausdorff:
            hd = calcular_hausdorff(img1_path, img2_path)
            resultado["Hausdorff"] = round(hd, 4)

        if usar_psnr:
            psnr_val = calcular_psnr(imagem1, imagem2)
            resultado["PSNR"] = round(psnr_val, 4)

        resultados.append(resultado)

    return resultados

def imprimir_resultados(resultados, usar_ssim_rmse, usar_snr, usar_hausdorff, usar_psnr):
    if not resultados:
        print("Nenhuma imagem valida encontrada.")
        return

    colunas = ["Imagem"]
    if usar_ssim_rmse:
        colunas += ["SSIM", "RMSE"]
    if usar_snr:
        colunas += ["SNR"]
    if usar_hausdorff:
        colunas += ["Hausdorff"]
    if usar_psnr:
        colunas += ["PSNR"]

    print("\n--- Resultados ---")
    print(" | ".join(colunas))
    print("-" * 60)

    for r in resultados:
        linha = [r["nome"]]
        if usar_ssim_rmse:
            linha.append(str(r.get("SSIM", "-")))
            linha.append(str(r.get("RMSE", "-")))
        if usar_snr:
            linha.append(str(r.get("SNR", "-")))
        if usar_hausdorff:
            linha.append(str(r.get("Hausdorff", "-")))
        if usar_psnr:
            linha.append(str(r.get("PSNR", "-")))
        print(" | ".join(linha))

    # Médias
    print("\n--- Médias ---")
    soma = {k: 0 for k in colunas[1:]}
    for r in resultados:
        for k in soma:
            if k in r:
                soma[k] += r[k]
    for k in soma:
        media = soma[k] / len(resultados)
        print(f"{k}: {round(media, 4)}")

    # Primeira e última imagem
    print("\n--- Primeira imagem ---")
    print(resultados[0])
    print("\n--- Última imagem ---")
    print(resultados[-1])

def main():
    # Configurações fixas (ajuste conforme sua necessidade)
    args = {
        "perfeitas": "C:/Users/Arlison Gaspar/Desktop/Projetos Visual studio/Aprimoramento de imagens/LINT-projeto/Projeto-LINT/script/downgrade/blurred",  # Substitua pelo seu caminho
        "aprimoradas": "C:/Users/Arlison Gaspar/Desktop/Projetos Visual studio/Aprimoramento de imagens/LINT-projeto/Projeto-LINT/script/enhancement/scunet/blurred",
        "ssim_rmse": False,   # Ativa SSIM e RMSE
        "snr": False,        # Desativa SNR
        "hausdorff": True, # Desativa Hausdorff
        "psnr": True        # Ativa PSNR
    }

    pares = carregar_imagens(args["perfeitas"], args["aprimoradas"])
    resultados = aplicar_metricas(
        pares, args["ssim_rmse"], args["snr"], args["hausdorff"], args["psnr"]
    )
    imprimir_resultados(resultados, args["ssim_rmse"], args["snr"], args["hausdorff"], args["psnr"])

if __name__ == "__main__":
    main()