import cv2
import sys
from progress_bar import print_progress_bar

def video_roi(video_path, x, y, w, h):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 获取视频的总帧数和帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 获取原视频的帧率和FourCC编码
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    # fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(*cv2.VideoWriter_fourcc_decode(fourcc_code))
    original_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_chars = "".join([chr((original_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    fourcc = cv2.VideoWriter_fourcc(*fourcc_chars)

    file_extension = video_path.split('.')[-1]

    # 输出视频文件路径
    output_path = f"{video_path.rsplit('.', 1)[0]}_ROI({x},{y},{w},{h}).{file_extension}"

    # 创建视频写入对象
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

    print(f"Total Frames: {total_frames}")
    print(f"Frame Rate: {frame_rate} fps")
    print(f"Resolution: {frame_width} x {frame_height}")

    print(f"ROI Top-Left Corner: ({x}, {y}), ROI Size: {w}x{h}")

    # 计算并打印ROI中心点坐标和尺寸
    roi_center_x, roi_center_y = x + w // 2, y + h // 2
    print(f"ROI Center Coordinates: ({roi_center_x}, {roi_center_y})")
    print(f"ROI Size: {w}x{h}")


    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print_progress_bar(frame_count, total_frames, prefix=f'croping the viedo:', suffix='Complete')

        # 裁剪ROI区域
        roi_frame = frame[y:y+h, x:x+w]

        # 写入新视频
        out.write(roi_frame)

    # 释放资源
    cap.release()
    out.release()
    print(f"ROI video saved to {output_path}")


if __name__ == "__main__":
    # 读取视频文件
    # video_path = input("Enter the path of the video file: ")
    video_paths = sys.argv[1:]

    if not video_paths:
        print("No video file paths provided.")
        sys.exit()


    # cap = cv2.VideoCapture(video_path)
        
    # 第一个视频路径用于选择或输入ROI
    first_video_path = video_paths[0]
    cap = cv2.VideoCapture(first_video_path)

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
    cap.release()

    # video_roi(video_path, x, y, w, h)
    for video_path in video_paths:
        video_roi(video_path, x, y, w, h)
