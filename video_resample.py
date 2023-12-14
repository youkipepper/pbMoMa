from moviepy.editor import VideoFileClip
import os
import sys

def resample_video(input_video_path, output_frame_rate):
    # 加载视频文件
    clip = VideoFileClip(input_video_path)

    # 获取不包含扩展名的原始文件名
    file_name, file_extension = os.path.splitext(input_video_path)
    new_file_name = f"{file_name}_{output_frame_rate}fps{file_extension}"

    # 设置新的帧率
    new_clip = clip.set_fps(output_frame_rate)

    # 保存新视频到一个新文件
    new_clip.write_videofile(new_file_name)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python video_resample.py <input_video_path> <new_frame_rate>")
        sys.exit(1)

    input_video = sys.argv[1]
    new_frame_rate = int(sys.argv[2])  # 将命令行参数转换为整数
    resample_video(input_video, new_frame_rate)
