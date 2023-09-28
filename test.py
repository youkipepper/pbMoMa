import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 读取视频文件
video_path = 'media/input_video_01.mp4'
video_filename = os.path.splitext(os.path.basename(video_path))[0]  # 提取视频文件名
cap = cv2.VideoCapture(video_path)

# 获取视频的总帧数和帧率
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Total Frames: {total_frames}")
print(f"Frame Rate: {frame_rate} fps")

# 选择感兴趣区域（ROI）
print("Select a ROI and then press SPACE or ENTER button!")
print("Cancel the selection process by pressing c button!")

while True:
    ret, first_frame = cap.read()
    if not ret:
        break

    cv2.imshow("Select ROI", first_frame)
    roi = cv2.selectROI("Select ROI", first_frame, fromCenter=False)
    cv2.destroyWindow("Select ROI")

    if roi[2] > 0 and roi[3] > 0:  # 确保选择了有效的ROI
        break

# 获取感兴趣区域的坐标
x, y, w, h = map(int, roi)

# 初始化追踪点（固定x坐标，选择的y坐标）
point_of_interest = np.array([[x + w // 2, y + h // 2]], dtype=np.float32)
p0 = point_of_interest

# 创建一个空的y坐标列表
y_tracks = []

try:
    frame_count = 0  # 用于计算帧数的变量

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 打印帧数信息
        frame_count += 1
        print(f"Processing Frame {frame_count} / {total_frames}")

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用Canny边缘检测来提取边缘信息
        edges = cv2.Canny(gray_frame, 50, 150)

        # 获取感兴趣区域的内容
        roi_frame = edges[y:y+h, x:x+w]

        # 寻找边缘上的点
        nonzero_indices = np.nonzero(roi_frame)
        if len(nonzero_indices[1]) > 0:
            # 获取y坐标值
            y_value = nonzero_indices[0][0] + y  # 加上y坐标偏移量
            y_tracks.append(y_value)

            # 实时绘制边缘点
            frame_with_edge_point = frame.copy()
            for x_edge, y_edge in zip(nonzero_indices[1], nonzero_indices[0]):
                cv2.circle(frame_with_edge_point, (x_edge + x, y_edge + y), 1, (0, 255, 0), -1)
            
            # 在预览窗口中显示带有边缘点的视频
            cv2.imshow("Edge Points", frame_with_edge_point)
            cv2.waitKey(1)

except KeyboardInterrupt:
    pass

finally:
    # 关闭视频文件
    cap.release()
    cv2.destroyAllWindows()

# 在完成数据收集后打印y_tracks
print("Collected Y Tracks:")
print(y_tracks)

# 创建保存图像的文件夹（如果不存在）
output_folder = os.path.join('fig', video_filename)
os.makedirs(output_folder, exist_ok=True)

# interpolated_y_tracks = []
# interpolation_factor = 4  # 插值因子

# for i in range(len(y_tracks) - 1):
#     for j in range(interpolation_factor):
#         interpolated_value = y_tracks[i] + (y_tracks[i + 1] - y_tracks[i]) * j / interpolation_factor
#         interpolated_y_tracks.append(interpolated_value)

# 增加图像大小和曲线点数
plt.figure(figsize=(14, 6), dpi=100)
plt.plot(y_tracks, linewidth=1.5)
plt.xlabel('Track Index')
plt.ylabel('Displacement (Y-coordinate)')
plt.title('Displacement vs. Track Index')
plt.tight_layout()  # 自动调整布局以适应图像大小

# 保存图像到本地
displacement_vs_track_index_filename = os.path.join(output_folder, f'{video_filename}_displacement_vs_track_index.png')
plt.savefig(displacement_vs_track_index_filename, dpi=300)  # 保存图像时指定更高的dpi

# 提取振动信息
def extract_vibration_info(track, sampling_rate):
    # 检查轨迹是否为空
    if len(track) == 0:
        return [], []

    # 计算傅立叶变换
    N = len(track)
    fft_result = np.fft.fft(track)

    # 计算频率
    freqs = np.fft.fftfreq(N, 1 / sampling_rate)

    # 计算幅度
    amplitude = np.abs(fft_result)

    # 排除直流分量
    freqs_without_dc = freqs[1:]
    amplitude_without_dc = amplitude[1:]

    return freqs_without_dc, amplitude_without_dc

# 设置采样频率
sampling_rate = frame_rate

# 提取振动信息
freqs, amplitudes = extract_vibration_info(y_tracks, sampling_rate)

# # 绘制频谱图
# plt.figure(figsize=(10, 4))
# plt.plot(freqs, amplitudes)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.title('Vibration Spectrum')

# # 保存图像到本地
# vibration_spectrum_filename = os.path.join(output_folder, f'{video_filename}_vibration_spectrum.png')
# plt.savefig(vibration_spectrum_filename)

# 增加图像大小和曲线点数
plt.figure(figsize=(14, 6), dpi=100)
plt.plot(freqs, amplitudes, linewidth=1.5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Vibration Spectrum')
plt.tight_layout()  # 自动调整布局以适应图像大小

# 保存图像到本地
vibration_spectrum_filename = os.path.join(output_folder, f'{video_filename}_vibration_spectrum.png')
plt.savefig(vibration_spectrum_filename, dpi=300)  # 保存图像时指定更高的dpi

# 显示图像
plt.show()
