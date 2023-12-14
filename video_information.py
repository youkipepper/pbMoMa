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

# Calculate the total video duration in seconds
total_duration = total_frames / frame_rate

print(f"Total Frames: {total_frames}")
print(f"Frame Rate: {frame_rate} fps")
print(f"Resolution: {frame_width} x {frame_height}")
print(f"Total Video Duration: {total_duration} seconds")
