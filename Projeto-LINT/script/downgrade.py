import cv2
import os
import numpy as np

if __name__ == "__main__":
    vid_name = 'teste_virat_dataset'
    path_images = f'cut/{vid_name}/'
    pre_imgs = os.listdir(path_images)
    img = []

    for i in pre_imgs:
        i = path_images + i
        img.append(i)

        for i in range(len(img)):
            imgread = cv2.imread(img[i], cv2.IMREAD_UNCHANGED)

            # Desfoque gaussiano
            """
            blur = cv2.GaussianBlur(imgread, (15, 15), cv2.BORDER_DEFAULT)
            cv2.imwrite(f'downgrade/blur{i}.png', blur)
            """
            
            # Resolução menor
            """
            scale = 50
            width = int(imgread.shape[1] * scale / 100)
            height = int(imgread.shape[0] * scale / 100)
            resized = cv2.resize(imgread, (width, height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f'teste/resize{i}.png', resized)
            """

            # Ruído salt and pepper
            noise_prob = 0.05
            width = imgread.shape[1]
            height = imgread.shape[0]
            noise_mask = np.random.rand(height, width)
            salt = noise_mask < noise_prob / 2
            pepper = noise_mask > 1 - noise_prob / 2
            imgread[salt] = np.random.randint(200, 256, size=np.count_nonzero(salt))
            imgread[pepper] = np.random.randint(0, 51, size=np.count_nonzero(pepper))
            cv2.imwrite(f'teste/noise{i}.png', imgread)

    print("Frames alterados copiados para o caminho 'downgrade/'")
