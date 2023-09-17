import cv2

def show_video(input_file):
    cap = cv2.VideoCapture(input_file)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow("Video", frame)

        # 设置按键监听，按下"q"键退出循环
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_file = 'media/input_video_01.mp4'  # 输入视频文件名

    show_video(input_file)
