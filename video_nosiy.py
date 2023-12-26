import numpy as np
import cv2
import sys
from progress_bar import print_progress_bar

# def add_noise(image, noise_type="gaussian"):
#     """
#     Add specified noise to an image.
#     :param image: input image
#     :param noise_type: type of noise to add ('gaussian', 'salt_pepper', 'poisson', 'speckle', etc.)
#     :return: noisy image
#     """
#     row, col, ch = image.shape
#     if noise_type == "gaussian":
#         mean = 0
#         var = 0.2
#         sigma = var**0.5
#         gauss = np.random.normal(mean, sigma, (row, col, ch))
#         gauss = gauss.reshape(row, col, ch)
#         noisy = image + gauss
#         return noisy.clip(0, 255).astype(np.uint8)
#     elif noise_type == "salt_pepper":
#         s_vs_p = 0.2
#         amount = 0.02
#         out = np.copy(image)
#         # Salt mode
#         num_salt = np.ceil(amount * image.size * s_vs_p)
#         coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
#         out[tuple(coords)] = 1
#         # Pepper mode
#         num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
#         coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
#         out[tuple(coords)] = 0
#         return out
#     elif noise_type == "poisson":
#         vals = len(np.unique(image))
#         vals = 2 ** np.ceil(np.log2(vals))
#         noisy = np.random.poisson(image * vals) / float(vals)
#         return noisy
#     elif noise_type == "speckle":
#         gauss = np.random.randn(row, col, ch) * 0.1
#         gauss = gauss.reshape(row, col, ch)        
#         noisy = image + image * gauss
#         return noisy.clip(0, 255).astype(np.uint8)

#     return image  # Return original image if noise type is not recognized

def add_noise(image, noise_type="gaussian", noise_level=0.1):
    """
    Add specified noise to an image with a given noise level.
    :param image: input image
    :param noise_type: type of noise to add ('gaussian', 'salt_pepper', 'poisson', 'speckle', etc.)
    :param noise_level: intensity of the noise
    :return: noisy image
    """
    row, col, ch = image.shape
    if noise_type == "gaussian":
        mean = 0
        var = noise_level
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy.clip(0, 255).astype(np.uint8)
    elif noise_type == "salt_pepper":
        s_vs_p = 0.5
        amount = noise_level
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 1
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0
        return out
    elif noise_type == "poisson":
        # Poisson noise level is not easily adjustable, it depends on the image data
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        gauss = np.random.randn(row, col, ch) * noise_level
        gauss = gauss.reshape(row, col, ch)        
        noisy = image + image * gauss
        return noisy.clip(0, 255).astype(np.uint8)

    return image  # Return original image if noise type is not recognized

def video_noisy(video_path, noise_type, noise_level):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取原视频的帧率和FourCC编码
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    original_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_chars = "".join([chr((original_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    fourcc = cv2.VideoWriter_fourcc(*fourcc_chars)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    file_extension = video_path.split('.')[-1]
    output_path = f"{video_path.rsplit('.', 1)[0]}_{noise_type}_{noise_level}.{file_extension}"

    # 创建视频写入对象
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print_progress_bar(frame_count, total_frames, prefix=f'Adding noise to the video:', suffix='Complete')

        # 添加噪声
        noisy_frame = add_noise(frame, noise_type, noise_level)

        # 写入新视频
        out.write(noisy_frame.astype(np.uint8))

        cv2.imshow("Nosiy frame", noisy_frame)
        if cv2.waitKey(11) & 0xFF == ord('q'):
                break 

    # 释放资源
    cap.release()
    out.release()
    print(f"Noisy video saved to {output_path}")

    return output_path

if __name__ == "__main__":
    video_path = '/Users/youkipepper/Desktop/pbMoMa/media/test/test_cropped_1-11.mp4'
    nosiy_video_path = video_noisy(video_path, 'salt_pepper', 0.05)