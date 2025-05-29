import cv2
import os
import numpy as np
from tqdm import tqdm  # Para barra de progresso (opcional)

def create_output_folders(base_output_path):
    """Cria pastas de saída para cada tipo de downgrade"""
    folders = {
        'blur': 'blurred',
        'resize': 'resized',
        'noise': 'noisy'
    }
    
    for folder in folders.values():
        os.makedirs(os.path.join(base_output_path, folder), exist_ok=True)
    
    return folders

def apply_blur(image, kernel_size=(15, 15)):
    """Aplica desfoque gaussiano"""
    return cv2.GaussianBlur(image, kernel_size, cv2.BORDER_DEFAULT)

def apply_resize(image, scale_percent=50):
    """Reduz a resolução da imagem"""
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

def apply_noise(image, noise_prob=0.05):
    """Adiciona ruído salt-and-pepper"""
    # Converter para float para operações
    noisy = image.astype(np.float32)
    
    # Gerar máscaras de ruído
    salt_mask = np.random.rand(*image.shape[:2]) < noise_prob/2
    pepper_mask = np.random.rand(*image.shape[:2]) > 1 - noise_prob/2
    
    # Aplicar ruído
    noisy[salt_mask] = 255  # Salt (branco)
    noisy[pepper_mask] = 0  # Pepper (preto)
    
    return noisy.astype(np.uint8)

def process_images(input_path, output_path):
    """Processa todas as imagens JPG no diretório de entrada"""
    # Criar pastas de saída
    folders = create_output_folders(output_path)
    
    # Listar apenas arquivos JPG
    image_files = [f for f in os.listdir(input_path) 
                  if f.lower().endswith(('.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Nenhuma imagem JPG encontrada em {input_path}")
        return
    
    print(f"Processando {len(image_files)} imagens...")
    
    for filename in tqdm(image_files):
        try:
            # Caminhos completos
            input_file = os.path.join(input_path, filename)
            base_name = os.path.splitext(filename)[0]
            
            # Ler imagem
            img = cv2.imread(input_file)
            if img is None:
                print(f"\nErro ao ler {filename}")
                continue
            
            # Aplicar downgrades
            blurred = apply_blur(img)
            resized = apply_resize(img)
            noisy = apply_noise(img)
            
            # Salvar resultados
            cv2.imwrite(os.path.join(output_path, folders['blur'], f"{base_name}_blur.jpg"), blurred)
            cv2.imwrite(os.path.join(output_path, folders['resize'], f"{base_name}_small.jpg"), resized)
            cv2.imwrite(os.path.join(output_path, folders['noise'], f"{base_name}_noisy.jpg"), noisy)
            
        except Exception as e:
            print(f"\nErro ao processar {filename}: {str(e)}")
            continue

if __name__ == "__main__":
    # Configurações 
    input_folder = r'Projeto-LINT/script/imagens'
    output_folder = r'Projeto-LINT/script/downgrade'
    
    # Verificar se a pasta de entrada existe
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Pasta de entrada não encontrada: {input_folder}")
    
    # Processar imagens
    process_images(input_folder, output_folder)
    
    print("Processamento concluído! Imagens salvas em:")
    print(f"- Desfocadas: {os.path.join(output_folder, 'blurred')}")
    print(f"- Reduzidas: {os.path.join(output_folder, 'resized')}")
    print(f"- Com ruído: {os.path.join(output_folder, 'noisy')}")