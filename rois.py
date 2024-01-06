# notation 获取视频中多个roi位置信息

import cv2

def select_rois(video_path):
    # 读取视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    # 读取第一帧作为背景图像
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video.")
        cap.release()
        return []

    cv2.namedWindow("Select ROIs", cv2.WINDOW_NORMAL)  # 可能有助于窗口在macOS上正常显示
    rois = []  # 用于存储所有ROI的列表

    # 选择多个ROI
    while True:
        selected_rois = cv2.selectROIs("Select ROIs", frame, fromCenter=False, showCrosshair=True)
        if len(selected_rois) == 0:
            break

        for roi in selected_rois:
            x, y, w, h = roi
            rois.append((x, y, w, h))
        break

    cap.release()
    cv2.destroyAllWindows()
    return rois

if __name__ == "__main__":
    video_path = input("Enter the path of the video file: ")
    selected_rois = select_rois(video_path)
    print("Selected ROIs:")
    for i, roi in enumerate(selected_rois):
        print(f"ROI {i+1}: Position (x, y) = ({roi[0]}, {roi[1]}), Size (width, height) = ({roi[2]}, {roi[3]})")
