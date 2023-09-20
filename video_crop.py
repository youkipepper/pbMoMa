import cv2
import os

def crop_video(input_file, output_file, crop_by_time=True):
    # 打开视频文件
    cap = cv2.VideoCapture(input_file)

    # 检查视频文件是否成功打开
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 在窗口上选择裁剪区域
    cv2.namedWindow("Video Cropper")
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        cap.release()
        cv2.destroyAllWindows()
        return

    (x, y, w, h) = cv2.selectROI("Video Cropper", frame, fromCenter=False)
    cv2.destroyAllWindows()

    # 开始读取视频并裁剪
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(5), (w, h))

    if crop_by_time:
        start_time = float(input("请输入开始时间（秒）："))
        end_time = float(input("请输入结束时间（秒）："))
    else:
        start_time = 0
        end_time = float('inf')  # 无穷大，即按照原视频长度剪裁

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 获取当前帧的时间
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        # 在指定时间范围内进行裁剪
        if start_time <= current_time <= end_time:
            # 裁剪视频帧
            cropped_frame = frame[y:y+h, x:x+w]

            # 写入裁剪后的帧到输出视频
            out.write(cropped_frame)

        # 如果超过结束时间，则停止裁剪
        if current_time > end_time:
            break

    # 释放视频资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("视频画面裁剪完成")


if __name__ == "__main__":
    input_file = 'media/shihumobile.mp4'    # 输入视频文件名

    # 获取输入文件的原始文件名和扩展名
    file_name, file_ext = os.path.splitext(os.path.basename(input_file))

    # 拼接新的文件名
    output_file = os.path.join('media', f'{file_name}_cut{file_ext}')

    crop_by_time = input("是否按照时间范围剪裁？(y/n): ").strip().lower() == 'y'
    crop_video(input_file, output_file, crop_by_time)
