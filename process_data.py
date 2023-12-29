import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from scipy.signal import find_peaks

from AMPD import AMPD

import matplotlib.ticker as ticker

marked_points = []
plt.rcParams.update({'font.size': 14})

# def on_click(event, freqs, amplitudes):
#     """处理点击事件，标记并存储最近的峰值点"""
#     click_freq = event.xdata
#     if click_freq is None:  # 忽略图外的点击
#         return
#     peaks, _ = find_peaks(amplitudes)
#     nearest_peak = peaks[np.abs(freqs[peaks] - click_freq).argmin()]

#     marked_point = (freqs[nearest_peak], amplitudes[nearest_peak])
#     marked_points.append(marked_point)

#     plt.plot(marked_point[0], marked_point[1], 'ro')
#     plt.text(marked_point[0], marked_point[1], f'({marked_point[0]:.2f}, {marked_point[1]:.2f})')
#     plt.draw()


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


def extract_vibration_info(track, sampling_rate):
    if len(track) == 0:
        return [], []
    N = len(track)
    fft_result = np.fft.fft(track)
    freqs = np.fft.fftfreq(N, 1 / sampling_rate)
    amplitude = np.abs(fft_result)
    freqs_without_dc = freqs[1:]
    amplitude_without_dc = amplitude[1:]
    return freqs_without_dc, amplitude_without_dc


def process_csv_data(csv_file_path, sampling_rate):
    # 读取CSV文件
    data = pd.read_csv(csv_file_path, header=None)

    # 计算原始数据中0的数量
    original_zero_count = (data == 0).sum().sum()

    # 替换0值为NaN
    data.replace(0, np.nan, inplace=True)

    # 计算替换后NaN的数量（之前为0的数量）
    replaced_zero_count = data.isna().sum().sum()

    # 打印原始数据中0的数量和被替换的0的数量
    print(f"Original dataset had {original_zero_count} zeros.")
    print(f"Replaced {replaced_zero_count} zeros with NaN in the dataset.")

    # 删除包含NaN的行
    data.dropna(inplace=True)

    y_tracks = data[0].tolist()

    # 获取CSV文件的基本名称
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]

    # 创建保存图像的文件夹（如果不存在）
    output_folder = os.path.join("/Users/youkipepper/Desktop/pbMoMa/fig", base_name)
    os.makedirs(output_folder, exist_ok=True)

    time_array = np.arange(len(y_tracks)) / sampling_rate


    # 生成时程图
    plt.figure(figsize=(14, 6), dpi=100)
    # plt.figure(figsize=(6, 5), dpi=100)

    # plt.xlim(0, max(time_array))
    # plt.xlim(30, 60)
    # plt.ylim(1030, 1040)

    # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(4))
    # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.plot(y_tracks, linewidth=1.5)
    plt.plot(time_array, y_tracks, linewidth=1.5)
    # plt.xlabel("Track Index")
    plt.xlabel("Time (s)", labelpad=15)
    # plt.ylabel("Displacement")
    plt.ylabel("Displacement (Px)", labelpad=15)
    plt.ylabel("Voltage (mV)", labelpad=15)
    
    plt.title(f"Displacement")

    plt.tight_layout()
    displacement_vs_track_index_filename = os.path.join(
        output_folder, f"{base_name}_displacement_vs_track_index.png"
    )
    plt.savefig(displacement_vs_track_index_filename, dpi=300)


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
    ax.set_title("Vibration Spectrum")
    plt.tight_layout()

    # 保存图像的文件名，您需要根据需要修改
    vibration_spectrum_filename = os.path.join(
        output_folder, f"{base_name}_vibration_spectrum.png"
    )
    plt.savefig(vibration_spectrum_filename, dpi=300)

    mark_choice = input("Do u wanna mark the point? (y/n)")
    if mark_choice == "y":
        plt.show()

        # 重新绘制图像以包含标注的点
        fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
        ax.plot(positive_freqs, positive_amplitudes, linewidth=1.5)
        for point in marked_points:
            ax.plot(point[0], point[1], "ro")
            ax.text(point[0], point[1], f"({point[0]:.5f}, {point[1]:.2f})")

        ax.set_xlabel("Frequency (Hz)", fontsize=14, labelpad=15)
        ax.set_ylabel("Amplitude", fontsize=14, labelpad=15)

        # ax.set_title("Vibration Spectrum with Marked Peaks")
        plt.tight_layout()

        vibration_spectrum_marked_filename = os.path.join(
            output_folder, f"{base_name}_vibration_spectrum_marked.png"
        )
        plt.savefig(vibration_spectrum_marked_filename, dpi=300)

        selected_freqs = [abs(point[0]) for point in marked_points]  # 转换为绝对值
        unique_freqs = set(selected_freqs)  # 去除重复值
        sorted_freqs = sorted(unique_freqs)  # 排序
        return sorted_freqs
    else:
        # 自动标记最大的峰值点
        fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
        ax.plot(positive_freqs, positive_amplitudes, linewidth=1.5)
        mark_highest_peak(positive_freqs, positive_amplitudes, ax)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Vibration Spectrum with Highest Peak Marked")
        plt.tight_layout()

        vibration_spectrum_marked_filename = os.path.join(
            output_folder, f"{base_name}_vibration_spectrum_highest_peak.png"
        )
        plt.savefig(vibration_spectrum_marked_filename, dpi=300)

    ######
    # # 生成频谱图
    # plt.figure(figsize=(14, 6), dpi=100)
    # plt.plot(freqs, amplitudes, linewidth=1.5)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude')
    # plt.title(f'Vibration Spectrum')
    # plt.tight_layout()
    # vibration_spectrum_filename = os.path.join(output_folder, f'{base_name}_vibration_spectrum.png')
    # plt.savefig(vibration_spectrum_filename, dpi=300)
    ######

    # plt.show()


if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print("Usage: python process_csv.py <csv_file_path> <sampling_rate>")
    #     sys.exit(1)
    # csv_file_path = sys.argv[1]

    csv_file_path = input('please input csv_file_path: ')

    sampling_rate = 60

    process_csv_data(csv_file_path, sampling_rate)
