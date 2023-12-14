import os
import shutil

def copy_files_including_all_keywords(source_folder, destination_folder, keywords, file_extension):
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历源文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 检查文件名是否包含所有关键词，并且具有指定的扩展名
            if all(keyword in file for keyword in keywords) and file.endswith(file_extension):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(destination_folder, file)

                # 复制文件
                shutil.copy2(source_file, destination_file)
                print(f"Copied: {source_file} to {destination_file}")

# 使用示例
source_folder = '/Users/youkipepper/Desktop/pbMoMa/fig'  # 替换为源文件夹的路径
destination_folder = '/Users/youkipepper/Desktop/pbMoMa/paper_fig'  # 替换为目标文件夹的路径
keywords = ['231130_01']  # 替换为文件名必须同时包含的关键词列表
file_extension = '.png'  # 替换为文件扩展名，例如 '.txt'

copy_files_including_all_keywords(source_folder, destination_folder, keywords, file_extension)
