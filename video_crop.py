# notation func：视频剪辑

import cv2
import os
from video_rotate import rotate_video, select_two_points, calculate_rotation_angle
from progress_bar import print_progress_bar

def video_info_and_cut(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return None

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = fourcc_code

    print(f"Video Dimensions: {frame_width}x{frame_height}")
    print(f"Total Frames: {total_frames}")
    print(f"Frame Rate: {fps} fps")
    print(f"Duration: {duration} seconds")
    print(f"FourCC Code: {fourcc_code}")

    # 获取裁剪时间
    start_time = input("Enter start time in seconds (or leave blank to start from beginning): ")
    end_time = input("Enter end time in seconds (or leave blank to end at the video's end): ")

    start_frame = int(float(start_time) * fps) if start_time else 0
    end_frame = int(float(end_time) * fps) if end_time else total_frames

    start_label = start_time if start_time else "start"
    end_label = end_time if end_time else "end"

    # 设置临时输出文件路径
    base_name, ext = os.path.splitext(os.path.basename(video_path))
    output_path_temp = os.path.join(os.path.dirname(video_path), f"{base_name}_temp{ext}")

    # 创建视频写入对象
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path_temp, fourcc, fps, (width, height))

    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret or current_frame >= end_frame:
            break

        if current_frame >= start_frame:
            out.write(frame)

        current_frame += 1
        print_progress_bar(current_frame, total_frames, prefix='Cropping video:', suffix='Complete', length=50)

    cap.release()
    out.release()

    # 确定是否替换原文件
    replace_original = input("Do you want to replace the original file? (y/n): ").strip().lower()
    if replace_original == 'y':
        os.replace(output_path_temp, video_path)
        return video_path
    else:
        # 设置输出文件路径，包含开始时间和结束时间
        output_path_final = os.path.join(os.path.dirname(video_path), f"{base_name}_cropped_{start_label}-{end_label}{ext}")
        os.rename(output_path_temp, output_path_final)
        return output_path_final

if __name__ == "__main__":
    video_path = input("Enter the path of the video file: ")
    result_path = video_info_and_cut(video_path)
    if result_path:
        print(f"Video has been processed and saved at: {result_path}")

