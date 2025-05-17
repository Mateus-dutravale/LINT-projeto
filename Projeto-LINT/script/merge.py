import cv2
import os

if __name__ == "__main__":
    vid_name = 'teste_virat_dataset'
    path_images = 'downgrade/'
    pre_imgs = os.listdir(path_images)
    img = []

    for i in pre_imgs:
        i = path_images + i
        img.append(i)

    # Tamanho do vídeo original
    readImg = cv2.imread(img[0], cv2.IMREAD_UNCHANGED)
    height = readImg.shape[0]
    width = readImg.shape[1]
    frameSize = (width, height)

    # Frames por segundo do vídeo original
    original_vid_path = f'source/{vid_name}.mp4'
    original_vid = cv2.VideoCapture(original_vid_path)
    fps = original_vid.get(cv2.CAP_PROP_FPS)

    output_video_path = f'merge/MERGED_{vid_name}' + '.mp4'
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, frameSize) # *'MJPG'

    for i in range(len(img)): 
        video.write(cv2.imread(img[i]))

    video.release()
    print(f"Vídeo 'MERGED_{vid_name}' criado no diretório 'merge/'")