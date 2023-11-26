import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

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
    output_folder = os.path.join('/Users/youkipepper/Desktop/pbMoMa/fig', base_name)
    os.makedirs(output_folder, exist_ok=True)

    # 生成时程图
    plt.figure(figsize=(14, 6), dpi=100)
    plt.plot(y_tracks, linewidth=1.5)
    plt.xlabel('Track Index')
    plt.ylabel('Displacement')
    plt.title(f'Displacement vs. Track Index')
    plt.tight_layout()
    displacement_vs_track_index_filename = os.path.join(output_folder, f'{base_name}_displacement_vs_track_index.png')
    plt.savefig(displacement_vs_track_index_filename, dpi=300)

    # 提取振动信息
    freqs, amplitudes = extract_vibration_info(y_tracks, sampling_rate)

    # 生成频谱图
    plt.figure(figsize=(14, 6), dpi=100)
    plt.plot(freqs, amplitudes, linewidth=1.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'Vibration Spectrum')
    plt.tight_layout()
    vibration_spectrum_filename = os.path.join(output_folder, f'{base_name}_vibration_spectrum.png')
    plt.savefig(vibration_spectrum_filename, dpi=300)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_csv.py <csv_file_path> <sampling_rate>")
        sys.exit(1)
        
    
    csv_file_path = sys.argv[1]
    sampling_rate = float(sys.argv[2])
    process_csv_data(csv_file_path, sampling_rate)
