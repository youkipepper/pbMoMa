import cv2

def crop_video(input_file, output_file):
    # 打开视频文件
    cap = cv2.VideoCapture(input_file)

    # 检查视频文件是否成功打开
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 创建一个窗口并显示视频帧
    cv2.namedWindow("Video Cropper")
    cv2.imshow("Video Cropper", cv2.imread(input_file))

    # 在窗口上选择裁剪区域
    (x, y, w, h) = cv2.selectROI("Video Cropper", cap, fromCenter=False)
    cv2.destroyAllWindows()

    # 开始读取视频并裁剪
    cap.release()
    cap = cv2.VideoCapture(input_file)
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(5), (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 裁剪视频帧
        cropped_frame = frame[y:y+h, x:x+w]

        # 写入裁剪后的帧到输出视频
        out.write(cropped_frame)

    # 释放视频资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("视频画面裁剪完成")

if __name__ == "__main__":
    input_file = 'media/input_video_01.mp4'    # 输入视频文件名
    output_file = 'media/output.mp4'  # 输出视频文件名

    crop_video(input_file, output_file)
