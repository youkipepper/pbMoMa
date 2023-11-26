import cv2
import os
import sys
import subprocess
from xls2csv import xls2csv
from process_data import process_csv_data
from frequency import frequency
from phasebasedMoMag import phaseBasedMagnify

from video_rotate import rotate_video, select_two_points, calculate_rotation_angle

from gray_level_edge import gray_level_detct

# 使用命令行参数作为输入路径
input_path = input("Enter the path of the video file: ")

# 询问用户是否需要旋转视频
rotate_choice = input("Do you want to rotate the video? (y/n): ").strip().lower()
if rotate_choice == 'y':
    cap = cv2.VideoCapture(input_path)
    points = select_two_points(cap)
    if points:
        angle = calculate_rotation_angle(points)
        input_path = rotate_video(input_path, angle)  # 更新input_path为旋转后的视频路径

        gray_detct_choice = input("Do you wana use gray_level_detct.py? (y/n): ").strip().lower()
        if gray_detct_choice == 'y':
            csv_path=gray_level_detct(input_path, 20, 40, 30, 0)
            base_name =os.path.splitext(os.path.basename(csv_path))[0]
            process_csv_data(csv_path, 25)

if input_path.endswith('.xls'):
    csv_file_path=xls2csv(input_path)
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    process_csv_data(csv_file_path, 100, base_name)

else:
    # 读取视频文件
    cap = cv2.VideoCapture(input_path)

    choice = input("Press 'M' to manually select ROI or 'E' to enter ROI center and size: ").strip().upper()

    if choice == 'M':
        # 选择感兴趣区域（ROI）
        print("Select a ROI and then press SPACE or ENTER button!")
        print("Cancel the selection process by pressing c button!")
        cap.release()  # 关闭视频文件

        while True:
            cap = cv2.VideoCapture(input_path)  # 重新打开视频文件
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
        cap.release()  # 关闭视频文件
        cap = cv2.VideoCapture(input_path)  # 重新打开视频文件
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

    
    # 提取放大前模态信息
    frequency(input_path, x, y, w, h)

    # 设置视频源、输出文件名、最大帧数等参数
    vidFname = input_path
    maxFrames = 20000
    windowSize = 30
    factor = 20
    fpsForBandPass = 25
    lowFreq = 4
    highFreq = 15
    # vidFnameOut = vidFname + '-Mag%dIdeal-lo%d-hi%d.avi' % (factor, lowFreq, highFreq)

    # 构建输出文件名
    vidFnameOut = os.path.join('media_mag', os.path.basename(vidFname).replace('.mp4', '-Mag%dIdeal-lo%d-hi%d.avi' % (factor, lowFreq, highFreq)))

    # 开始运动放大
    mag_vid_path = phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq, x, y, w, h)
    
    # 可选：将输出视频转换为MP4格式
    print("\nStart convert video to mp4\n")
    subprocess.call(['ffmpeg', '-i', mag_vid_path, mag_vid_path.replace('.avi', '.mp4')])

    # 提取放大后模态信息
    frequency(mag_vid_path, x, y, w, h)