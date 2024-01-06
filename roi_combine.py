# notation func：roi拼接

import cv2
import numpy as np
import os
from progress_bar import print_progress_bar  # 假设这是从您已有的 progress_bar.py 文件中导入的函数

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = ''.join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
    cap.release()
    return total_frames, fps, size, codec

def select_roi_mode():
    mode = input("选择ROI模式: 输入 'manual' 手动选择 或者 'input' 输入坐标和尺寸: ")
    return mode

def get_roi(video_path, mode):
    if mode == 'manual':
        # 这里需要一个图形界面来选择ROI，暂未实现
        pass
    elif mode == 'input':
        roi_input = input("一次性输入ROI的坐标和尺寸（格式：x,y,width,height）: ")
        x, y, w, h = map(int, roi_input.split(','))
        return (x, y, w, h)
    else:
        print("无效的模式")
        return None

def process_video(video_path, roi, frame_range):
    cap = cv2.VideoCapture(video_path)
    images = []
    total_frames = frame_range[1] - frame_range[0] + 1
    for frame_num in range(frame_range[0], frame_range[1] + 1):
        print_progress_bar(frame_num - frame_range[0], total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            roi_image = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
            images.append(roi_image)
        else:
            break
    cap.release()
    return np.hstack(images) if images else None

def main():
    video_path = input("输入视频路径: ")
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # 从路径中提取视频名称
    total_frames, fps, size, codec = get_video_info(video_path)
    print(f"视频信息: 总帧数={total_frames}, 帧率={fps}, 尺寸={size}, 编码格式={codec}")

    roi_mode = select_roi_mode()
    roi = get_roi(video_path, roi_mode)
    if roi is None:
        return

    start_frame = int(input("输入开始帧数: "))
    end_frame = int(input("输入结束帧数: "))
    frame_range = (max(0, start_frame), min(total_frames, end_frame))

    result_image = process_video(video_path, roi, frame_range)
    if result_image is not None:
        output_folder = "pic"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_filename = f"{video_name}_output.png"  # 使用视频名称作为文件名的一部分
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, result_image)
        print(f"结果图片已保存到 {output_path}")
    else:
        print("处理视频时发生错误")

if __name__ == "__main__":
    main()
