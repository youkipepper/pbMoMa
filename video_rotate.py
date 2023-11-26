import cv2
import numpy as np
import os
from progress_bar import print_progress_bar

def select_two_points(cap):
    points = []
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            if len(points) == 2:
                cv2.line(frame, points[0], points[1], (255, 0, 0), 2)
            cv2.imshow('Frame', frame)

    ret, frame = cap.read()
    if not ret:
        return None

    cv2.imshow('Frame', frame)
    cv2.setMouseCallback('Frame', click_event)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 2:
        print("Two points were not selected")
        return None

    return points

def calculate_rotation_angle(points):
    pt1, pt2 = points
    angle = np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))
    return angle

def rotate_video(video_path, angle):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    height, width = frame.shape[:2]

    # 获取原视频的 FourCC 编码
    original_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_chars = "".join([chr((original_fourcc >> 8 * i) & 0xFF) for i in range(4)])

    # 计算旋转参数
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated = cv2.warpAffine(frame, rotation_matrix, (width, height))

    # 设置输出视频参数
    video_dir = os.path.dirname(video_path)
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(video_dir, f"{video_filename}_rotated_{angle:.0f}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*fourcc_chars)
    out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rotated = cv2.warpAffine(frame, rotation_matrix, (width, height))
        out.write(rotated)

        current_frame += 1
        print_progress_bar(current_frame, total_frames, prefix='Rotating video:', suffix='Complete', length=50)

    cap.release()
    out.release()
    print("\nRotation completed. Processed video saved as:", out_path)

    return out_path

if __name__ == "__main__":
    video_path = input("Enter the path of the video file: ")
    cap = cv2.VideoCapture(video_path)

    points = select_two_points(cap)
    if points:
        angle = calculate_rotation_angle(points)
        rotate_video(video_path, angle)
