import cv2
import numpy as np
import pandas as pd
from darkest_edge import darkest_edge_detection

from progress_bar import print_progress_bar

import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def select_roi(frame):
    rois = []
    temp_frame = frame.copy()
    cv2.imshow("Frame", temp_frame)

    while True:
        roi = cv2.selectROI("Frame", temp_frame, fromCenter=False, showCrosshair=True)
        if roi[2] != 0 and roi[3] != 0:  # 确保ROI有效
            rois.append(roi)
            x, y, w, h = roi
            cv2.rectangle(temp_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("Frame", temp_frame)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # 按下Esc键退出
            break

    cv2.destroyAllWindows()
    return rois

def video_edge(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Total Frames: {total_frames}, Frame Rate: {frame_rate}, Resolution: {frame_width} x {frame_height}")

    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    rois = select_roi(first_frame)

    # 计算ROI的最小开始列号和最大结束列号
    start_col = min(x for x, _, w, _ in rois)
    end_col = max(x + w for x, _, w, _ in rois)


    y_matrix = np.zeros((total_frames, frame_width))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Finished reading video at frame index {frame_count} (Total reported frames: {total_frames}).")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_with_edges = frame.copy()

        for roi in rois:
            x, y, w, h = roi
            roi_frame = gray_frame[y:y+h, x:x+w]
            edge_points = darkest_edge_detection(roi_frame, 6)

            # print(f"Frame {frame_count}: Detected {len(edge_points)} edge points in ROI {roi}")  # 调试信息

            for col, y_val in edge_points:
                global_col = x + col
                if 0 <= global_col < frame_width:
                    y_matrix[frame_count, global_col] = y_val + y
                    # 标记检测到的边缘
                    cv2.line(frame_with_edges, (global_col, int(y_val + y)), (global_col, int(y_val + y)), (0, 255, 0), 3)

        cv2.imshow("Edges", frame_with_edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        print_progress_bar(frame_count, total_frames, prefix='Extract modal information:', suffix='Complete')
    cap.release()
    print(f"ROI Column Range: Start - {start_col}, End - {end_col}, Total Columns in Range - {frame_width}")
    return y_matrix, start_col, end_col

def model(y_matrix, start_col, end_col, sample_rate, frequency):
    amplitude_by_col = []

    for col in range(start_col, end_col):
        # 提取y_matrix中对应列的非零数据
        column_data = y_matrix[:, col]
        nonzero_data = column_data[column_data != 0]

        # print(f"Column {col}: Extracted {len(nonzero_data)} non-zero points")

        # 检查非零数据点的数量
        if len(nonzero_data) == 0:
            amplitude_by_col.append(0)
            continue

        # 傅立叶变换
        N = len(nonzero_data)
        yf = fft(nonzero_data)
        xf = fftfreq(N, 1 / sample_rate)


        # 找到最接近输入频率的频率索引
        freq_index = np.argmin(np.abs(xf - frequency))
        amplitude = np.abs(yf[freq_index])
        amplitude_by_col.append(amplitude)

    # 绘制振型图像
    plt.figure(figsize=(10, 6))
    plt.plot(range(start_col, end_col), amplitude_by_col)
    plt.xlabel('Column Number')
    plt.ylabel('Amplitude')
    plt.title('Cable Vibration Mode Shape')
    plt.tight_layout()  # 自动调整布局以适应图像大小
    plt.grid(False)

    # 保存图像
    plt.savefig('vibration_mode_shape.png', dpi=300)
    plt.show()

    return amplitude_by_col


if __name__ == "__main__":
    video_path = input('please input video path: ')
    y_matrix, start_col, end_col = video_edge(video_path)
    if y_matrix is not None:
        # 将每行作为CSV文件中的一行
        pd.DataFrame(y_matrix).to_csv('output_matrix.csv', header=False, index=False)
    
    model(y_matrix, start_col, end_col, 9.75 , 100)

