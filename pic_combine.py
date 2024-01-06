# notation func: 图像拼接

import cv2
import os
from PIL import Image
import numpy as np

def create_video(image_folder, video_name="output_video.mp4", fps=5):
    # 获取所有的png图像文件
    files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    # 创建两个字典来存储两种类型的图像
    displacement_images = {}
    vibration_images = {}

    # 分类图像并存入相应的字典
    for file in files:
        number = file.split('(')[1].split(',')[0]  # 提取数字
        if "displacement_vs_track_index" in file:
            displacement_images[number] = file
        elif "vibration_spectrum_marked" in file:
            vibration_images[number] = file

    # 对数字进行排序
    sorted_numbers = sorted(displacement_images.keys())

    # 确定视频的分辨率
    sample_img = Image.open(os.path.join(image_folder, files[0]))
    width, height = sample_img.size

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用AV1编码
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height * 2))

    # 对每个数字，找到匹配的图像对并添加到视频
    for number in sorted_numbers:
        if number in vibration_images:
            img1_path = os.path.join(image_folder, displacement_images[number])
            img2_path = os.path.join(image_folder, vibration_images[number])
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            combined_img = Image.new('RGB', (width, height * 2))
            combined_img.paste(img1, (0, 0))
            combined_img.paste(img2, (0, height))

            # 转换为OpenCV格式并写入视频
            frame = cv2.cvtColor(np.array(combined_img), cv2.COLOR_RGB2BGR)
            video.write(frame)

    # 释放资源
    video.release()
    print(f"Video saved as {video_name}")

# 调用函数
create_video('/Users/youkipepper/Desktop/pbMoMa/fig/231129_02_cropped_2-32_darkest')
