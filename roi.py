import cv2

video_path = input("Enter the path of the video file: ")
cap = cv2.VideoCapture(video_path)

print("Select a ROI and then press SPACE or ENTER button!")

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

print(x, y, w, h)