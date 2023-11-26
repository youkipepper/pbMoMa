import sys
from PIL import Image
import os

def combine_images(image_paths):
    images = [Image.open(img_path.strip()) for img_path in image_paths]
    widths, heights = zip(*(i.size for i in images))

    # 计算总宽度和总高度
    max_width = max(widths) * 2
    total_height = sum(max(heights[i:i+2]) for i in range(0, len(images), 2))

    # 创建一个新的图像
    new_im = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for i in range(0, len(images), 2):
        x_offset = 0
        for img in images[i:i+2]:
            new_im.paste(img, (x_offset, y_offset))
            x_offset += img.size[0]

        y_offset += max(images[i:i+2], key=lambda x: x.size[1]).size[1]

    return new_im

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请使用命令行参数提供一个 txt 文件路径。")
        sys.exit(1)

    txt_file_path = sys.argv[1]
    with open(txt_file_path, 'r') as file:
        image_paths = file.readlines()

    if not image_paths:
        print(f"{txt_file_path} 文件中没有找到图片路径。")
    else:
        combined_image = combine_images(image_paths)
        combined_image.show()

        # 生成与 txt 文件同名的图片文件
        dir_name = os.path.dirname(txt_file_path)
        base_name = os.path.splitext(os.path.basename(txt_file_path))[0]
        output_path = os.path.join(dir_name, base_name + ".jpg")
        combined_image.save(output_path)
        print(f"图片已保存为 {output_path}")