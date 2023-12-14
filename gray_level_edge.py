import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import csv

from progress_bar import print_progress_bar

# 定义全局变量来存储选中的点
global point_selected, point
point_selected = False
point = (0, 0)

def select_point(event, x, y, flags, param):
    """
    鼠标回调函数，用于在图像上选择一个点。
    """
    global point, point_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True

def calc_comp_level(gray_statistics, min_value):
    """
    计算给定灰度统计数据的级别值。
    """
    break_num = 0
    continue_length = 0
    candidate_lengths = []
    for value in gray_statistics:
        if value > min_value:
            break_num = 0
            continue_length += 1
        else:
            if continue_length != 0:
                break_num += 1
                if break_num == 2:
                    candidate_lengths.append(continue_length)
                    continue_length = 0
                    break_num = 0
                else:
                    continue_length += 1
    max_num = max(candidate_lengths) if candidate_lengths else 0
    return max_num

def EGSDI(ROI):
    """
    处理图像的感兴趣区域（ROI），计算灰度统计和主导灰度值。
    """
    m, n = ROI.shape
    Gray_statistics = np.zeros(256)
    I_mean = np.mean(ROI)
    for k in range(m):
        for j in range(n):
            Gray_statistics[ROI[k, j]] += 1
    min_value = 2 / (m * n)
    Gray_statistics /= (m * n)
    max_num = calc_comp_level(Gray_statistics, min_value)
    power = round(np.log2(max_num))
    compre_level = min(8, 2 ** power)
    compres_graySta = np.zeros(256 // compre_level)
    for k in range(256 // compre_level):
        compres_graySta[k] = np.sum(Gray_statistics[k * compre_level:(k + 1) * compre_level])
    expect, max_level = np.max(compres_graySta), np.argmax(compres_graySta)
    temp_i = firstMoment(Gray_statistics, compre_level, max_level)
    CompLevel_Imean = int(np.ceil(I_mean / compre_level))
    max_area = 0
    level2 = 0
    if temp_i > I_mean:
        i1 = temp_i
        for k in range(CompLevel_Imean - 1, 0, -1):
            if max_area <= compres_graySta[k]:
                max_area = compres_graySta[k]
                level2 = k
        i0 = firstMoment(Gray_statistics, compre_level, level2)
    else:
        i0 = temp_i
        level2 = CompLevel_Imean
        for k in range(CompLevel_Imean, 256 // compre_level):
            if max_area <= compres_graySta[k]:
                max_area = compres_graySta[k]
                level2 = k
        i1 = firstMoment(Gray_statistics, compre_level, level2)
    return i0, i1, I_mean

def firstMoment(Gray_statistics, compre_level, now_level):
    """
    计算给定灰度统计数据的首选矩。
    """
    Expect = 0
    prob = 0
    for i in range(1, compre_level + 1):
        grayscale = np.ceil((now_level - 1) * compre_level + i)
        Expect += grayscale * Gray_statistics[int(grayscale)]
        prob += Gray_statistics[int(grayscale)]
    temp_i = Expect / prob if prob != 0 else 0
    m_test = round(temp_i)
    if compre_level / 2 + 2 < temp_i < 256 - compre_level / 2 - 2:
        Expect = 0
        prob = 0
        for i in range(1, compre_level + 5):
            grayscale = np.ceil(m_test - compre_level / 2 - 2 + i)
            Expect += grayscale * Gray_statistics[int(grayscale)]
            prob += Gray_statistics[int(grayscale)]
        intensity = Expect / prob if prob != 0 else 0
    else:
        intensity = temp_i
    return intensity

def geo_proc(s1, s_0, alpha):
    """
    根据提供的参数执行几何计算。
    """
    pi = np.pi
    s = np.sin((pi / 180) * alpha)
    c = np.cos((pi / 180) * alpha)
    t = np.tan((pi / 180) * alpha)
    k = t / 2

    d=0

    if s1 <= k * s_0 and s1 > 0:
        d = np.sqrt(s1 * 2 * s * c)
    elif s1 == 0:
        d = 0
    elif k * s_0 < s1 <= s_0 * (1 - k):
        d = c * (2 * (s1 / np.sqrt(s_0)) + (np.sqrt(s_0) * s / c)) / 2
    elif s_0 * (1 - k) < s1 <= s_0:
        d = c * (np.sqrt(s_0) * (1 + s / c) - np.sqrt(2 * (s_0 - s1) * s / c))
    elif s1 == s_0:
        d = np.sqrt(s_0) * (c + s)
    return d

def MotionDetect(video_path, p_x, p_y, width, height, alpha, n):
    video = cv2.VideoCapture(video_path)
    frame_number = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    s_0 = width * height
    record_d1 = np.zeros(frame_number)

    for i in range(frame_number):
        ret, img = video.read()
        if not ret:
            break

        # 更新进度条
        print_progress_bar(i + 1, frame_number, prefix='Progress:', suffix='Complete', length=50)

        # 处理ROI并计算运动量
        ROI = img[p_y - height // 2:p_y + height // 2, p_x - width // 2:p_x + width // 2, 0]
        i0, i1, I_mean = EGSDI(ROI)
        s1 = s_0 * (I_mean - i0) / (i1 - i0) if i1 != i0 else 0
        d1 = geo_proc(s1, s_0, alpha)
        record_d1[i] = d1

        # 在帧上绘制ROI矩形
        cv2.rectangle(img, (p_x - width // 2, p_y - height // 2), (p_x + width // 2, p_y + height // 2), (0, 255, 0), 2)

        # 计算文本显示的位置
        text_x = p_x - width // 2
        text_y = p_y - height // 2 - 10  # 在ROI框的上方显示文本

        # 将计算值作为文本放在视频帧上
        cv2.putText(img, f'i0: {i0:.2f}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f'i1: {i1:.2f}', (text_x, text_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f'I_mean: {I_mean:.2f}', (text_x, text_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f's0: {s_0}', (text_x, text_y - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f's1: {s1:.2f}', (text_x, text_y - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f'd1: {d1:.2f}', (text_x, text_y - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 显示帧
        cv2.imshow('Motion Tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    # 保存 record_d1 到 CSV 文件
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_folder_path = os.path.join('/Users/youkipepper/Desktop/pbMoMa/csv', f"{video_name}_gray")
    os.makedirs(csv_folder_path, exist_ok=True)
    csv_file_path = os.path.join(csv_folder_path, f"{video_name}_record_d1_{p_x}_{p_y}_{width}_{height}.csv")


    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for value in record_d1:
            writer.writerow([value])

    return csv_file_path


def gray_level_detct(video_path, width, height, alpha , n):
    # 打开视频文件
    # video_path = '/Users/youkipepper/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/728672f7f419fd286c097bb1bfa7096d/Message/MessageTemp/b4637eebf303696d0091212c5c573fd4/File/wusun.MP4'  # 示例视频路径
    video = cv2.VideoCapture(video_path)

    # 获取视频的第一帧
    ret, frame = video.read()
    if not ret:
        print("无法读取视频文件")
        return
    
    choice = input("Press 'M' to manually select ROI or 'P' to manually select ROI center_point ").strip().upper()

    if choice == 'M':

        # 显示第一帧并等待用户选择点
        cv2.namedWindow("Frame")
        # cv2.setMouseCallback("Frame", select_point)
        # while not point_selected:
        #     cv2.imshow("Frame", frame)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        roi = cv2.selectROI(frame, False)

        cv2.destroyAllWindows()

        # # 使用用户选定的点
        # p_x, p_y = point

        if any(roi):
            # ROI的坐标和尺寸
            x, y, width, height = roi

            # 计算ROI的中心点坐标
            p_x = x + width // 2
            p_y = y + height // 2

    elif choice == 'P':
        # 用户选择点选中心点
        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", select_point)
        global point_selected, point
        point_selected = False
        point = (0, 0)

        while not point_selected:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        # 使用用户选定的点
        p_x, p_y = point

    else:
        print("无效选择")
        return

    csv_file_path = MotionDetect(video_path, p_x, p_y, width, height, alpha, n)
    print(f"Record_d1 saved to CSV at: {csv_file_path}")

    return csv_file_path


if __name__ == "__main__":
    video_path = '/Users/youkipepper/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/728672f7f419fd286c097bb1bfa7096d/Message/MessageTemp/b4637eebf303696d0091212c5c573fd4/File/wusun.MP4'  # 示例视频路径
    gray_level_detct(video_path, 15, 15, 0, 0)

