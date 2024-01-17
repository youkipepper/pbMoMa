import os
import re
import shutil

def organize_files(target_dir):
    # 正则表达式用于匹配特定格式的文件名
    pattern = re.compile(r"(\d{6}_[^_]+)(?:_|$)")

    # 用于存储找到的符合条件的文件名及其新路径
    files_to_move = {}

    # 遍历目标文件夹中的所有项
    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        # 确保这是一个文件
        if os.path.isfile(item_path):
            match = pattern.match(item)
            if match:
                new_folder_name = match.group(1)
                new_folder_path = os.path.join(target_dir, new_folder_name)
                files_to_move[item_path] = new_folder_path

    # 移动文件
    for old_path, new_path in files_to_move.items():
        # 如果新文件夹不存在，则创建
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        # 移动文件
        shutil.move(old_path, new_path)

    print("Files organized.")

# 使用示例
target_directory = "/Users/youkipepper/Desktop/pbMoMa/csv"  # 将这里替换为您的目标文件夹路径
organize_files(target_directory)
