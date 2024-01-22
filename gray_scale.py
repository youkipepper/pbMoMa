import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import io


def find_peaks(histogram, num_peaks=2, max_range= 80):
    """
    在直方图中查找前 num_peaks 个最大峰值。
    返回一个包含峰值位置的列表。
    """
    peaks = []
    for _ in range(num_peaks):
        peak = np.argmax(histogram)
        peaks.append(peak)
        # 将当前峰值附近的值设为零，以查找下一个峰值
        start = max(0, peak - max_range)
        end = min(len(histogram), peak + max_range)
        for i in range(start, end):
            histogram[i] = 0
    return peaks


def fill_small_non_mark_areas(color_image, n):
    # 定义红色
    color = [0, 255, 0]

    # 获取图像的高度和宽度
    height, width = color_image.shape[:2]

    for col in range(width):
        start = None  # 非红色区间的开始
        for row in range(height):
            # 检查像素是否为红色
            if not np.array_equal(color_image[row, col], color):
                if start is None:
                    start = row  # 开始新的非红色区间
            else:
                if start is not None:
                    # 检查非红色区间的长度
                    if row - start < n:
                        # 将小于10的非红色区间填充为红色
                        color_image[start:row, col] = color
                    start = None  # 重置非红色区间的开始

        # 检查并填充最后一个区间（如果需要）
        if start is not None and height - start < 10:
            color_image[start:height, col] = color


# def process_image_areas(color_image, threshold, mark_color=[0, 255, 0], fill_color=[255, 255, 255]):
#     # 获取图像的高度和宽度
#     height, width = color_image.shape[:2]

#     for col in range(width):
#         start = None  # 区间的开始
#         is_marked = False  # 标记区间是否是标记颜色

#         for row in range(height):
#             # 检查像素是否为标记颜色
#             if np.array_equal(color_image[row, col], mark_color):
#                 if start is None:
#                     start = row
#                     is_marked = True
#             else:
#                 if start is None:
#                     start = row
#                     is_marked = False
#                 elif is_marked:
#                     # 检查标记颜色区间的长度
#                     if row - start < threshold:
#                         # 将小于阈值的标记颜色区间去掉
#                         color_image[start:row, col] = fill_color
#                     start = None

#         # 检查并处理最后一个区间（如果需要）
#         if start is not None and is_marked and height - start < threshold:
#             color_image[start:height, col] = fill_color

#     return color_image
            
# def process_image_areas(color_image, threshold, mark_color=[0, 0, 255], fill_color=[255, 255, 255]):
#     height, width = color_image.shape[:2]

#     for col in range(width):
#         start = None
#         mark_count = 0  # 用于记录满足条件的mark_color区间数量

#         for row in range(height):
#             # 如果当前像素是mark_color
#             if np.array_equal(color_image[row, col], mark_color):
#                 if start is None:  # 区间的开始
#                     start = row
#             else:
#                 if start is not None:  # 区间的结束
#                     # 检查mark_color区间的长度
#                     if row - start >= threshold:
#                         mark_count += 1
#                         # 仅保留第二个符合条件的区间
#                         if mark_count != 2:
#                             color_image[start:row, col] = fill_color
#                     else:
#                         # 如果区间长度小于阈值，则用fill_color填充
#                         color_image[start:row, col] = fill_color
#                     start = None

#         # 处理最后一个区间（如果需要）
#         if start is not None:
#             if height - start < threshold or mark_count != 1:
#                 color_image[start:height, col] = fill_color

#     return color_image
            

def process_image_areas(color_image, threshold, n=-1, mark_color=[0, 0, 255], fill_color=[255, 255, 255]):
    height, width = color_image.shape[:2]

    for col in range(width):
        start = None
        mark_count = 0
        last_valid_start = None
        last_valid_end = None

        for row in range(height):
            if np.array_equal(color_image[row, col], mark_color):
                if start is None:
                    start = row  # 区间开始
            else:
                if start is not None:  # 区间结束
                    if row - start >= threshold:
                        mark_count += 1
                        if n == -1:
                            last_valid_start = start  # 更新最后一个有效区间的开始位置
                            last_valid_end = row  # 更新最后一个有效区间的结束位置
                        elif mark_count != n:
                            color_image[start:row, col] = fill_color  # 用fill_color填充
                    else:
                        color_image[start:row, col] = fill_color
                    start = None

        # 处理列的最后一个区间
        if start is not None:
            if height - start >= threshold:
                mark_count += 1
                if n == -1:
                    last_valid_start = start
                    last_valid_end = height
                elif mark_count != n:
                    color_image[start:height, col] = fill_color
            else:
                color_image[start:height, col] = fill_color

        # 如果n为-1，处理最后一个有效区间
        if n == -1:
            # 先将整列用fill_color填充
            color_image[:, col] = fill_color
            # 保留最后一个有效区间
            if last_valid_start is not None and last_valid_end is not None:
                color_image[last_valid_start:last_valid_end, col] = mark_color

    return color_image



def generate_gray_scale_histogram(image, peak_choice="2", keep_area = -1, fill_bug = 0, peak_range= 100):
    # 读取图像并转换为灰度图像
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("无法读取图像")
        return
    

    # 计算灰度值的频率分布
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    # 查找直方图中的最大峰值和第二大峰值
    peaks = find_peaks(histogram.copy(), num_peaks=2, max_range= peak_range)
    # print(f"max_range: {peak_range}, gray_level_1: {peaks[0]}, gray_level_2: {peaks[1]}, threshold: {(peaks[0]+peaks[1])//2}")

    # # 计算极值点的中点
    # x_centre = int((peaks[0] + peaks[1]) / 2)

    # # 二值化图像
    # _, binary_image = cv2.threshold(image, x_centre, 255, cv2.THRESH_BINARY)

    # 创建一个空的彩色图像
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    peak_max = max(peaks[0], peaks[1])
    peak_min = min(peaks[0], peaks[1])

    gray_threshold = None

    if peak_choice == "1":
        # 标记第一峰值邻域内的像素
        peak = peak_min
        lower_bound = 0
        upper_bound = peak +(peak_max-peak_min)//2
    elif peak_choice == "2":
        peak = peak_max
        lower_bound = peak -(peak_max-peak_min)//2
        upper_bound = 255
    elif peak_choice == "otsu":
        peak = None
        lower_bound = None
        upper_bound = None
        gray_threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        peak = peaks[0]
        lower_bound = max(0, peak - 50)
        upper_bound = min(255, peak + 50)
    # print(lower_bound, upper_bound)
    if peak_choice == "otsu":
        mask = image >= gray_threshold
    else:        
        mask = (image >= lower_bound) & (image <= upper_bound)
    color_image[mask] = [0, 255, 0]  # 将符合条件的像素点设为绿色（BGR格式） 
    
    if fill_bug != 0:
        fill_small_non_mark_areas(color_image, fill_bug)

    if keep_area != 0:
        process_image_areas(color_image, 10, keep_area, mark_color=[0, 255, 0], fill_color=[255, 255, 255])

    # 统计红色像素点的数量
    pixel_mask = (color_image == [0, 255, 0]).all(axis=2)
    s_1 = np.sum(pixel_mask)   


    # 统计大于和小于 x_centre 的像素点个数
    # s_1 = np.sum(mask)
    # s_2 = np.sum(binary_image > x_centre)

    # 计算所需值
    ratio = s_1 / image.shape[1]
    image_width_half = image.shape[1] // 2

    # # 修改原图中大于 x_centre 的像素点为红色
    # color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # color_image[binary_image < x_centre] = [0, 255, 0]  # BGR格式

    # 绘制灰度统计图
    plt.figure(figsize=(6, 4))
    plt.title('Gray Scale Histogram')
    plt.xlabel('Gray Level')
    plt.ylabel('Frequency')
    plt.plot(histogram)
    plt.scatter(peaks, [histogram[peak] for peak in peaks], color='red')  # 标记峰值
    plt.xlim([0, 256])

    plt.gca().axes.get_yaxis().set_visible(False)

    # 将图形保存到BytesIO对象中
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # 使用PIL打开图像并转换为OpenCV图像格式
    pil_img = Image.open(buf)
    hist_image = np.array(pil_img)
    hist_image = cv2.cvtColor(hist_image, cv2.COLOR_RGB2BGR)

    plt.close()
    buf.close()

    y_value = image.shape[0] - ratio

    if keep_area == -1:
        return [(image_width_half, y_value)], color_image, hist_image
    else:
        return [(image_width_half, ratio)], color_image, hist_image


def darkest_gray(image):
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for col in range(image.shape[1]):
        min_row = np.argmin(image[:, col])  # 找到该列的最小灰度值所在行
        color_image[min_row:, col] = [0, 255, 0]   
    pixel_mask = (color_image == [0, 255, 0]).all(axis=2)
    s_1 = np.sum(pixel_mask)
    ratio = s_1 / image.shape[1]
    image_width_half = image.shape[1] // 2
    y_value = image.shape[0] - ratio
    return [(image_width_half, y_value)], color_image

if __name__ == "__main__":
    # 使用示例
    image_path = input("input the path: ")  # 替换为您的图片路径
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread(image_path)
    ratio, color_image,hist_image = generate_gray_scale_histogram(image, "2", keep_area=-1, fill_bug=0, peak_range= 40)
    # ratio, color_image = darkest_gray(image)

    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"ret= {ret}")

    print(ratio)
    cv2.imshow("Color Image", color_image)
    cv2.imshow("Histogram Image", hist_image)
    cv2.imshow("OTSU", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

