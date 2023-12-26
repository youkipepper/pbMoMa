import cv2

video_path = input("Enter the path of the video file: ")
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Failed to read video")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 获取FourCC编码并转换为字符串格式
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

# 计算视频总时长（秒）
total_duration = total_frames / frame_rate

print(f"Total Frames: {total_frames}")
print(f"Frame Rate: {frame_rate} fps")
print(f"Resolution: {frame_width} x {frame_height}")
print(f"Total Video Duration: {total_duration} seconds")
print(f"Codec: {codec}")
