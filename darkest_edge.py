import numpy as np
from scipy.optimize import curve_fit

# 高斯函数定义
def gauss(x, a, b, c):
    return a * np.exp(-((x - b)**2) / (2 * c**2))

# 处理图像的函数
def darkest_edge_detection(img, n=6):
    rows, cols = img.shape
    max_points = []  # 存储每列中高斯拟合的最大值点坐标

    # 对每一列进行处理
    for col in range(cols):
        column_values = img[:, col]
        min_val_index = np.argmin(column_values)

        start_index = max(min_val_index - (n // 2), 0)
        end_index = min(min_val_index + (n // 2), rows)

        y_values = 255 - column_values[start_index:end_index]
        x_values = np.arange(start_index, end_index)

        # 高斯拟合
        try:
            params, _ = curve_fit(gauss, x_values, y_values, p0=[255, min_val_index, 1])
            a, b, _ = params
            max_points.append((col, b))  # 存储亚像素精度的y坐标
        except RuntimeError:
            continue

    return max_points

