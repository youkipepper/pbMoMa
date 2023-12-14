import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from scipy.signal import find_peaks
from darkest_edge import darkest_edge_detection
from progress_bar import print_progress_bar


marked_points = []

def mark_peaks(freqs, amplitudes, ax):
    peaks, _ = find_peaks(amplitudes)
    for peak in peaks:
        if freqs[peak] > 0:  # 确保频率大于0
            marked_point = (freqs[peak], amplitudes[peak])
            ax.plot(marked_point[0], marked_point[1], 'go')  # 使用绿色标记峰值点
            ax.text(marked_point[0], marked_point[1], f'({marked_point[0]:.2f}, {marked_point[1]:.2f})')

def mark_highest_peak(freqs, amplitudes, ax):
    peaks, _ = find_peaks(amplitudes)
    if peaks.size > 0:
        highest_peak = peaks[np.argmax(amplitudes[peaks])]  # 找到最大振幅的峰值
        if freqs[highest_peak] > 0:  # 确保频率大于0
            marked_point = (freqs[highest_peak], amplitudes[highest_peak])
            ax.plot(marked_point[0], marked_point[1], 'go')  # 使用绿色标记最大峰值点
            ax.text(marked_point[0], marked_point[1], f'({marked_point[0]:.2f}, {marked_point[1]:.2f})')

def on_click(event, freqs, amplitudes, ax, fig):
    """处理点击事件，标记或取消标记最近的峰值点"""
    click_freq = event.xdata
    if click_freq is None:  # 忽略图外的点击
        return

    # 左键点击选择点
    if event.button == 1:
        peaks, _ = find_peaks(amplitudes)
        nearest_peak = peaks[np.abs(freqs[peaks] - click_freq).argmin()]
        marked_point = (freqs[nearest_peak], amplitudes[nearest_peak])
        marked_points.append(marked_point)
        ax.plot(marked_point[0], marked_point[1], 'ro')
        ax.text(marked_point[0], marked_point[1], f'({marked_point[0]:.2f}, {marked_point[1]:.2f})')

    # 右键点击取消最近的选择
    elif event.button == 3:
        if marked_points:
            marked_points.pop()
            # 重绘图像以更新标记
            ax.clear()
            ax.plot(freqs, amplitudes, linewidth=1.5)
            for point in marked_points:
                ax.plot(point[0], point[1], 'ro')
                ax.text(point[0], point[1], f'({point[0]:.2f}, {point[1]:.2f})')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Vibration Spectrum')

    # 强制重新绘制图像
    fig.canvas.draw()


def frequency(video_path, x, y, w, h, edge_choice):
    video_filename = os.path.splitext(os.path.basename(video_path))[0]  # 提取视频文件名
    cap = cv2.VideoCapture(video_path)

    # 获取视频的总帧数和帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    y_matrix = np.full((frame_width, total_frames), np.nan)

    # 获取原视频的 FourCC 编码并转换为四字符的字符串
    fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_chars = "".join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])
    fourcc = cv2.VideoWriter_fourcc(*fourcc_chars)

    video_dir = os.path.dirname(video_path)
    out_path = os.path.join(video_dir, f"{video_filename}_{edge_choice}_edge_points_roi({int(x)},{int(y)},{w},{h}).mp4")
    out = cv2.VideoWriter(out_path, fourcc, frame_rate, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    print(f"Total Frames: {total_frames}")
    print(f"Frame Rate: {frame_rate} fps")
    print(f"Resolution: {frame_width} x {frame_height}")

    print(f"ROI Top-Left Corner: ({x}, {y}), ROI Size: {w}x{h}")

    # 计算并打印ROI中心点坐标和尺寸
    roi_center_x, roi_center_y = x + w // 2, y + h // 2
    print(f"ROI Center Coordinates: ({roi_center_x}, {roi_center_y})")
    print(f"ROI Size: {w}x{h}")

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
            # print(f"Processing Frame {frame_count} / {total_frames}")
            print_progress_bar(frame_count, total_frames, prefix=f'Extract modal information {edge_choice}:', suffix='Complete')

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 限定ROI
            roi_frame = gray_frame[y:y+h, x:x+w]
            edge_points = []  # 初始化 edge_points 变量

            if edge_choice == 'darkest' or edge_choice == 'darkest after amplified':
            # 使用最暗点边缘检测算法
                edge_points = darkest_edge_detection(roi_frame, 6)
            elif edge_choice == 'canny' or edge_choice == 'canny after amplified':                
            # use canny
                edge_image = cv2.Canny(roi_frame, 50, 150)
                y_coords, x_coords = np.where(edge_image == 255)
                edge_points = list(zip(x_coords, y_coords)) 
            # 处理ROI中心列的最大边缘点

            center_col = w // 2
            center_point = None
            for col, y_val in edge_points:
                if col == center_col:
                    y_tracks.append(y_val + y)  # 加上ROI的y坐标偏移
                    center_point = (x + w // 2, int(y_val) + y)  # 用于显示的整数坐标
                    break

            # 在预览窗口中显示带有中心边缘点的视频
            if center_point is not None:
                cv2.circle(frame, center_point, 4, (0, 0, 255), -1)

            out.write(frame)  # 将带有中心边缘点的帧写入新视频

            #show the video frames
            cv2.imshow("Edge Points", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 

    except KeyboardInterrupt:
        pass

    finally:
        # 关闭视频文件
        cap.release()
        cv2.destroyAllWindows()

    # 在完成数据收集后打印y_tracks
    print("Collected Y Tracks:")
    # print(y_tracks)

    # 视频处理循环结束后，将y_tracks写入CSV文件
    csv_output_folder = os.path.join('csv', f'{video_filename}_{edge_choice}')
    os.makedirs(csv_output_folder, exist_ok=True)
    csv_filename = os.path.join(csv_output_folder, f'{video_filename}_{edge_choice}_roi({int(x)},{int(y)},{w},{h})_y_tracks_center_x={roi_center_x}.csv')

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for y in y_tracks:
            csv_writer.writerow([y])

    print(f"Y tracks saved to CSV at: {csv_filename}")

    # 创建保存图像的文件夹（如果不存在）
    output_folder = os.path.join('fig', f'{video_filename}_{edge_choice}')
    os.makedirs(output_folder, exist_ok=True)

    # 增加图像大小和曲线点数
    plt.figure(figsize=(14, 6), dpi=100)
    plt.plot(y_tracks, linewidth=1.5)
    plt.xlabel('Track Index')
    plt.ylabel('Displacement (Y-coordinate)')
    plt.title(f'Displacement vs. Track Index ({edge_choice}) (x={x + w // 2})')
    plt.tight_layout()  # 自动调整布局以适应图像大小

    displacement_vs_track_index_filename = os.path.join(output_folder, f'{video_filename}_{edge_choice}_roi({int(x)},{int(y)},{w},{h})_displacement_vs_track_index_x={roi_center_x}.png')
    plt.savefig(displacement_vs_track_index_filename, dpi=300)

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

    positive_freqs = freqs[:len(freqs)//2]
    positive_amplitudes = amplitudes[:len(amplitudes)//2]

    global marked_points
    marked_points.clear()  # 清空标记点列表

    # 绘制频谱图并连接点击事件
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
    ax.plot(positive_freqs, positive_amplitudes, linewidth=1.5)
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event,  positive_freqs, positive_amplitudes, ax, fig))

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Vibration Spectrum ({edge_choice}_roi({int(x)},{int(y)},{w},{h}))')
    plt.tight_layout()

    # 显示并保存不带标记点的图像
    vibration_spectrum_filename = os.path.join(output_folder, f'{video_filename}_{edge_choice}_roi({int(x)},{int(y)},{w},{h})_vibration_spectrum.png')
    plt.savefig(vibration_spectrum_filename, dpi=300)

    mark_choice = input("Do u wanna mark the point? (y/n)")
    if mark_choice == "y":

        # 显示图像，等待窗口关闭
        plt.show()  

        # 重新绘制图像以包含标注的点
        fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
        ax.plot(positive_freqs, positive_amplitudes, linewidth=1.5)
        for point in marked_points:
            ax.plot(point[0], point[1], 'ro')
            ax.text(point[0], point[1], f'({point[0]:.2f}, {point[1]:.2f})')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Vibration Spectrum with Marked Peaks ({edge_choice}_roi({int(x)},{int(y)},{w},{h}))')
        plt.tight_layout()

        # 保存带标记点的图像
        vibration_spectrum_marked_filename = os.path.join(output_folder, f'{video_filename}_{edge_choice}_roi({int(x)},{int(y)},{w},{h})_vibration_spectrum_marked.png')
        plt.savefig(vibration_spectrum_marked_filename, dpi=300)

        selected_freqs = [abs(point[0]) for point in marked_points]  # 转换为绝对值
        unique_freqs = set(selected_freqs)  # 去除重复值
        sorted_freqs = sorted(unique_freqs)  # 排序
        return sorted_freqs 
    else:
        # 自动标记峰值点
        fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
        ax.plot(positive_freqs, positive_amplitudes, linewidth=1.5)
        mark_highest_peak(positive_freqs, positive_amplitudes, ax)

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Vibration Spectrum with Highest Marked Peaks ({edge_choice}_roi({int(x)},{int(y)},{w},{h}))')
        plt.tight_layout()

        # 保存带峰值标记点的图像
        vibration_spectrum_highest_peak_filename = os.path.join(output_folder, f'{video_filename}_{edge_choice}_roi({int(x)},{int(y)},{w},{h})_vibration_spectrum_marked.png')
        plt.savefig(vibration_spectrum_highest_peak_filename, dpi=300)

if __name__ == "__main__":
    # 读取视频文件
    video_path = input("Enter the path of the video file: ")
    cap = cv2.VideoCapture(video_path)

    choice = input("Press 'M' to manually select ROI or 'E' to enter ROI center and size: ").strip().upper()

    if choice == 'M':
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
                x, y, w, h = map(int, roi)
                break
    elif choice == 'E':
        # 用户输入ROI的左上角坐标和尺寸
        while True:
            try:
                x, y = map(int, input("Enter ROI top-left corner (x, y): ").split(','))
                w, h = map(int, input("Enter ROI size (width, height): ").split(','))
                # 确保ROI在图像范围内
                _, frame = cap.read()
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0] and x + w <= frame.shape[1] and y + h <= frame.shape[0]:
                    break
                else:
                    print("ROI is out of image bounds.")
            except ValueError:
                print("Invalid input. Please enter valid integers.")

    frequency(video_path, x, y, w, h, 'darkest')
    # selected_frequencies = frequency(video_path, x, y, w, h, 1)  
    # first_frequency = selected_frequencies[0]
    # print("Selected Frequencies:", selected_frequencies, "Hz")
    # print("First_frequency = ", first_frequency)


