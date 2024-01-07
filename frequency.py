import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from scipy.signal import find_peaks
from darkest_edge import darkest_edge_detection
from progress_bar import print_progress_bar
import matplotlib.ticker as ticker
from gray_scale import generate_gray_scale_histogram, darkest_gray

marked_points = []

def apply_noise(img, noise_type):
    if noise_type == "gaussian":
        mean = 0
        std_dev = 25
        noise = np.random.normal(mean, std_dev, img.shape)
        noisy_img = img + noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    elif noise_type == "salt_pepper":
        amount = 0.05
        out = np.copy(img)

        # Salt mode
        num_salt = np.ceil(amount * img.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        out[tuple(coords)] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        out[tuple(coords)] = 0

        return out

    elif noise_type == "uniform":
        noise = np.random.uniform(-50, 50, img.shape)
        noisy_img = img + noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    elif noise_type == "poisson":
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy_img = np.random.poisson(img * vals) / float(vals)
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    else:
        # No noise added
        return img


def mark_peaks(freqs, amplitudes, ax):
    peaks, _ = find_peaks(amplitudes)
    for peak in peaks:
        if freqs[peak] > 0:  # 确保频率大于0
            marked_point = (freqs[peak], amplitudes[peak])
            ax.plot(marked_point[0], marked_point[1], "go")  # 使用绿色标记峰值点
            ax.text(
                marked_point[0],
                marked_point[1],
                f"({marked_point[0]:.2f}, {marked_point[1]:.2f})",
            )


def mark_highest_peak(freqs, amplitudes, ax):
    peaks, _ = find_peaks(amplitudes)
    if peaks.size > 0:
        highest_peak = peaks[np.argmax(amplitudes[peaks])]  # 找到最大振幅的峰值
        if freqs[highest_peak] > 0:  # 确保频率大于0
            marked_point = (freqs[highest_peak], amplitudes[highest_peak])
            ax.plot(marked_point[0], marked_point[1], "go")  # 使用绿色标记最大峰值点
            ax.text(
                marked_point[0],
                marked_point[1],
                f"({marked_point[0]:.2f}, {marked_point[1]:.2f})",
            )


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
        ax.plot(marked_point[0], marked_point[1], "ro")
        ax.text(
            marked_point[0],
            marked_point[1],
            f"({marked_point[0]:.2f}, {marked_point[1]:.2f})",
        )

    # 右键点击取消最近的选择
    elif event.button == 3:
        if marked_points:
            marked_points.pop()
            # 重绘图像以更新标记
            ax.clear()
            ax.plot(freqs, amplitudes, linewidth=1.5)
            for point in marked_points:
                ax.plot(point[0], point[1], "ro")
                ax.text(point[0], point[1], f"({point[0]:.2f}, {point[1]:.2f})")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Vibration Spectrum")

    # 强制重新绘制图像
    fig.canvas.draw()


def frequency(video_path, x, y, w, h, edge_choice, noise_type=None, save_csv = False):
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
    grandparent_dir = os.path.dirname(os.path.dirname(video_dir))

    # 检查视频是否已在media_attached文件夹中
    if os.path.basename(video_dir) == "media_attached":
        out_dir = video_dir
    else:
        # 不在media_attached中，使用上上级目录的media_attached文件夹
        out_dir = os.path.join(grandparent_dir, "media_attached")
        # 如果目录不存在，则创建它
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # 构建输出视频的文件路径
    out_path = os.path.join(
        out_dir,
        f"{video_filename}_{edge_choice}_edge_roi({x},{y},{w},{h}).mp4",
    )


    out = cv2.VideoWriter(
        out_path,
        fourcc,
        frame_rate,
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    print(f"Total Frames: {total_frames}")
    print(f"Frame Rate: {frame_rate} fps")
    print(f"Resolution: {frame_width} x {frame_height}")

    print(f"ROI Top-Left Corner: ({x}, {y}), ROI Size: {w}x{h}")

    # 计算并打印ROI中心点坐标和尺寸
    roi_center_x, roi_center_y = x + w // 2, y + h // 2
    print(f"ROI Center Coordinates: ({roi_center_x}, {roi_center_y})")

    # 创建或检查media_attached文件夹
    media_attached_dir = os.path.join(os.getcwd(), "media_attached")
    if not os.path.exists(media_attached_dir):
        os.makedirs(media_attached_dir)

    # 构建灰度直方图视频的输出路径
    hist_video_path = os.path.join(
        media_attached_dir,
        f"{video_filename}_{edge_choice}_histogram_video.mp4",
    )

    # 创建灰度直方图视频输出
    hist_out = cv2.VideoWriter(
        hist_video_path,
        fourcc,
        frame_rate,
        (600, 400)
    )    

    # 初始化追踪点（固定x坐标，选择的y坐标）
    point_of_interest = np.array([[x + w // 2, y + h // 2]], dtype=np.float32)
    p0 = point_of_interest

    # 创建一个空的y坐标列表
    y_tracks = []

    # roi 信息
    text_offset_x = 10
    font_scale = 0.7
    font_thickness = 2
    line_spacing = 20

    color_frame = None

    try:
        frame_count = 0  # 用于计算帧数的变量

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 打印帧数信息
            frame_count += 1

            print_progress_bar(
                frame_count,
                total_frames,
                prefix=f"Extract modal information {edge_choice}:",
                suffix="Complete",
            )

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 限定ROI

            noisy_frame = apply_noise(gray_frame, noise_type)
            roi_frame = noisy_frame[y : y + h, x : x + w]

            # roi_frame = gray_frame[y:y+h, x:x+w]
            edge_points = []  # 初始化 edge_points 变量

            # notation 边缘提取算法
            if edge_choice == "darkest" or edge_choice == "darkest_amplified":
                # 使用最暗点边缘检测算法
                edge_points = darkest_edge_detection(
                    roi_frame, fit_type="gauss", degree=6
                )                 
                
            elif edge_choice == "canny" or edge_choice == "canny_amplified":
                # 使用 Canny 算法
                edge_image = cv2.Canny(roi_frame, 50, 150)
                edge_row_indices, edge_col_indices = np.where(edge_image == 255)
                edge_points = list(zip(edge_col_indices, edge_row_indices))

                # 存储每个边缘列的最大边缘行
                max_row_at_edge_col = {}
                for edge_col, edge_row in edge_points:
                    if edge_col not in max_row_at_edge_col or edge_row > max_row_at_edge_col[edge_col]:
                        max_row_at_edge_col[edge_col] = edge_row

                # 更新 edge_points 以包含每个边缘列的最大边缘行的点
                edge_points = [(edge_col, max_row) for edge_col, max_row in max_row_at_edge_col.items()]

            elif edge_choice == "gray" or edge_choice == "gray_amplified":
                edge_points, color_frame, hist_image =  generate_gray_scale_histogram(roi_frame, "1")
                frame[y:y+h, x:x+w] = color_frame
                # hist_out.write(hist_image)


            elif edge_choice == "dark_gray" or edge_choice == "dark_gray_amplified":
                edge_points, color_frame = darkest_gray(roi_frame)
                frame[y:y+h, x:x+w] = color_frame
            

            # notation 边缘线标注
            # for point in edge_points:
            #     roi_point = (int(point[0] + x), int(point[1] + y))
            #     # 在视频帧上用浅蓝色线条画出边缘点
            #     cv2.circle(frame, roi_point, 1, (255, 0, 0), -1)

            center_col = w // 2
            center_point = None
            for col, y_val in edge_points:
                if col == center_col:
                    y_tracks.append(y_val + y)  # 加上ROI的y坐标偏移
                    center_point = (x + w // 2, int(y_val) + y)  # 用于显示的整数坐标
                    break
            
            # 在预览窗口中显示带有中心边缘点的视频 # notation 显示中心边缘点
            if center_point is not None:
                cv2.circle(frame, center_point, 3, (0, 0, 255), -1)

            # out.write(frame)  # notation 只保存边缘点视频

            # 在视频帧上用绿色方框画出ROI区域
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 在视频帧上添加ROI信息的文本
            position_text = f"Position: ({x}, {y})"
            size_text = f"Size: {w}x{h}"

            text_y_position = (
                y - 2 * line_spacing
                if y - 2 * line_spacing > line_spacing
                else y + h + 2 * line_spacing
            )

            cv2.putText(
                frame,
                position_text,
                (x, text_y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                font_thickness,
            )
            cv2.putText(
                frame,
                size_text,
                (x, text_y_position + line_spacing + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                font_thickness,
            )

            # 绘制y_track值
            if y_tracks:  # 确保y_tracks不为空
                # y_track_text = f"y_track: {y_tracks[-1]:.5f}"  # 显示最新的y_track值
                y_track_text = f"y_track: {y_tracks[-1]:.3f}"  # 显示最新的y_track值

                # 为文本指定位置，放在ROI的右上外部
                text_position_x = x + w + text_offset_x
                text_position_y = y  # 设置为ROI的上边缘
                cv2.putText(
                    frame,
                    y_track_text,
                    (text_position_x, text_position_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 255),
                    font_thickness,
                )

            # out.write(frame) # notation 保存roi信息视频

            # show the video frames # notation 显示预览视频
            cv2.imshow("Edge Points", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    finally:
        # 关闭视频文件
        cap.release()
        hist_out.release()
        cv2.destroyAllWindows()

    # 在完成数据收集后打印y_tracks 
    # print("Collected Y Tracks:")
    # print(y_tracks)

    if save_csv == True:
        # 视频处理循环结束后，将y_tracks写入CSV文件
        # csv_output_folder = os.path.join("csv", f"{video_filename}_{edge_choice}")
        csv_output_folder = 'csv'
        os.makedirs(csv_output_folder, exist_ok=True)
        csv_filename = os.path.join(
            csv_output_folder,
            f"{video_filename}_{edge_choice}_roi({x},{y},{w},{h}).csv",
        )
        # print(f'y = {y} , int(y) = {int(y)}')

        with open(csv_filename, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            for track_y in y_tracks:
                csv_writer.writerow([track_y])

        print(f"Y tracks saved to CSV at: {csv_filename}")

    # 创建保存图像的文件夹（如果不存在）
    output_folder = os.path.join("fig", f"{video_filename}_{edge_choice}")
    os.makedirs(output_folder, exist_ok=True)
    # print(f'y = {y} , int(y) = {int(y)}')

    time_array = np.arange(len(y_tracks)) / frame_rate

    # 增加图像大小和曲线点数
    plt.figure(figsize=(14, 6), dpi=100)
    plt.xlim(0, max(time_array))
    plt.plot(time_array, y_tracks, linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (Px)")
    plt.title(f"Displacement vs. Track Index ({edge_choice}) (x={x + w // 2})")

    plt.tight_layout()  # 自动调整布局以适应图像大小

    displacement_vs_track_index_filename = os.path.join(
        output_folder,
        f"{video_filename}_{edge_choice}_roi({x},{y},{w},{h})_displacement_vs_track_index.png",
    )
    plt.savefig(displacement_vs_track_index_filename, dpi=300)
    # print(f'y = {y} , int(y) = {int(y)}')
    print(
        f"displacement_vs_track_index png saved at: {displacement_vs_track_index_filename}"
    )

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

    positive_freqs = freqs[: len(freqs) // 2]
    positive_amplitudes = amplitudes[: len(amplitudes) // 2]

    global marked_points
    marked_points.clear()  # 清空标记点列表

    # 绘制频谱图并连接点击事件
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
    ax.plot(positive_freqs, positive_amplitudes, linewidth=1.5)
    cid = fig.canvas.mpl_connect(
        "button_press_event",
        lambda event: on_click(event, positive_freqs, positive_amplitudes, ax, fig),
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Vibration Spectrum ({edge_choice}_roi({x},{y},{w},{h}))")
    plt.tight_layout()

    # 显示并保存不带标记点的图像
    vibration_spectrum_filename = os.path.join(
        output_folder,
        f"{video_filename}_{edge_choice}_roi({x},{y},{w},{h})_vibration_spectrum.png",
    )
    # plt.savefig(vibration_spectrum_filename, dpi=300)
    # print(f"vibration_spectrum png saved at: {vibration_spectrum_filename}")

    mark_choice = input("Do u wanna mark the point? (y/n)")
    if mark_choice == "y":
        # 显示图像，等待窗口关闭
        plt.show()

        # 重新绘制图像以包含标注的点
        fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
        ax.plot(positive_freqs, positive_amplitudes, linewidth=1.5)
        for point in marked_points:
            ax.plot(point[0], point[1], "ro")
            ax.text(point[0], point[1], f"({point[0]:.2f}, {point[1]:.2f})")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(
            f"Vibration Spectrum with Marked Peaks ({edge_choice}_roi({x},{y},{w},{h}))"
        )
        plt.tight_layout()

        # 保存带标记点的图像
        vibration_spectrum_marked_filename = os.path.join(
            output_folder,
            f"{video_filename}_{edge_choice}_roi({x},{y},{w},{h})_vibration_spectrum_marked.png",
        )
        plt.savefig(vibration_spectrum_marked_filename, dpi=300)
        print(
            f"vibration_spectrum_marked png saved at: {vibration_spectrum_marked_filename}"
        )

        selected_freqs = [abs(point[0]) for point in marked_points]  # 转换为绝对值
        unique_freqs = set(selected_freqs)  # 去除重复值
        sorted_freqs = sorted(unique_freqs)  # 排序
        return sorted_freqs
    else:
        # 自动标记峰值点
        fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
        ax.plot(positive_freqs, positive_amplitudes, linewidth=1.5)
        mark_highest_peak(positive_freqs, positive_amplitudes, ax)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(
            f"Vibration Spectrum with Highest Marked Peaks ({edge_choice}_roi({x},{y},{w},{h}))"
        )
        plt.tight_layout()

        # 保存带峰值标记点的图像
        vibration_spectrum_highest_peak_filename = os.path.join(
            output_folder,
            f"{video_filename}_{edge_choice}_roi({x},{y},{w},{h})_vibration_spectrum_marked.png",
        )
        plt.savefig(vibration_spectrum_highest_peak_filename, dpi=300)
        print(
            f"vibration_spectrum_highest_peak png saved at: {vibration_spectrum_highest_peak_filename}"
        )


if __name__ == "__main__":
    # 读取视频文件
    video_path = input("Enter the path of the video file: ")
    cap = cv2.VideoCapture(video_path)

    choice = (
        input("Press 'M' to manually select ROI or 'E' to enter ROI center and size: ")
        .strip()
        .upper()
    )

    if choice == "M":
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
    elif choice == "E":
        # 用户输入ROI的左上角坐标和尺寸
        while True:
            try:
                x, y = map(int, input("Enter ROI top-left corner (x, y): ").split(","))
                w, h = map(int, input("Enter ROI size (width, height): ").split(","))
                # 确保ROI在图像范围内
                _, frame = cap.read()
                if (
                    0 <= x < frame.shape[1]
                    and 0 <= y < frame.shape[0]
                    and x + w <= frame.shape[1]
                    and y + h <= frame.shape[0]
                ):
                    break
                else:
                    print("ROI is out of image bounds.")
            except ValueError:
                print("Invalid input. Please enter valid integers.")

    frequency(video_path, x, y, w, h, "gray") # notation 修改调试函数
    # selected_frequencies = frequency(video_path, x, y, w, h, 1)
    # first_frequency = selected_frequencies[0]
    # print("Selected Frequencies:", selected_frequencies, "Hz")
    # print("First_frequency = ", first_frequency)
