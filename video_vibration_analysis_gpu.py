import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 读取视频文件
video_path = 'media/video_input_01.avi'
cap = cv2.VideoCapture(video_path)

# 获取视频的总帧数和帧率
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Total Frames: {total_frames}")
print(f"Frame Rate: {frame_rate} fps")

# 初始化SIFT特征点检测器
sift = cv2.SIFT_create()

# 读取第一帧图像
ret, first_frame = cap.read()
gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# 在第一帧图像上绘制感兴趣区域的边界框
cv2.namedWindow("Select ROI")
roi = cv2.selectROI("Select ROI", first_frame, fromCenter=False)
cv2.destroyWindow("Select ROI")

# 获取感兴趣区域的坐标
x, y, w, h = roi

# 检测特征点和计算描述符
kp1, des1 = sift.detectAndCompute(gray_first_frame[y:y+h, x:x+w], None)

# 创建一个空的轨迹列表
tracks = [[] for _ in range(len(kp1))]

# 频域数据的采样率，根据视频帧率设置
sampling_rate = frame_rate

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

        # 检测特征点和计算描述符，仅在感兴趣区域内进行
        kp2, des2 = sift.detectAndCompute(gray_frame[y:y+h, x:x+w], None)

        # 使用FLANN匹配器匹配特征点
        flann = cv2.FlannBasedMatcher()
        matches = flann.knnMatch(des1, des2, k=2)

        # ... 其他代码 ...

except KeyboardInterrupt:
    pass

finally:
    # 关闭视频文件
    cap.release()

# 提取振动信息
def extract_vibration_info(track):
    # 假设轨迹数据是一维的
    y = np.array([point[1] for point in track])
    
    # 计算傅立叶变换
    N = len(y)
    fft_result = np.fft.fft(y)
    
    # 计算频率
    freqs = np.fft.fftfreq(N, 1 / sampling_rate)
    
    # 计算幅度
    amplitude = np.abs(fft_result)
    
    # 找到主频率和幅度
    main_frequency = freqs[np.argmax(amplitude)]
    main_amplitude = np.max(amplitude)
    
    return main_frequency, main_amplitude

# 用于存储振动信息的列表
vibration_info = []

# 对每个轨迹进行振动信息提取
for track in tracks:
    frequency, amplitude = extract_vibration_info(track)
    vibration_info.append((frequency, amplitude))

# 可视化振动信息
# 绘制频率和幅度图表
frequencies, amplitudes = zip(*vibration_info)

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(frequencies)
plt.xlabel('Track Index')
plt.ylabel('Frequency (Hz)')
plt.title('Vibration Frequencies')

plt.subplot(122)
plt.plot(amplitudes)
plt.xlabel('Track Index')
plt.ylabel('Amplitude')
plt.title('Vibration Amplitudes')

plt.tight_layout()

# 保存绘制的图片到指定文件夹下
save_dir = './fig'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Directory {save_dir} created.")
else:
    print(f"Directory {save_dir} already exists.")

save_path = os.path.join(save_dir, 'vibration_analysis_plot.png')
plt.savefig(save_path)
plt.show()

print(f"Vibration analysis plot saved to {save_path}")
