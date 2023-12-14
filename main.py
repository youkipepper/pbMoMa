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
from model import select_roi, video_edge, model

input_path = input("Enter the path of the file: ")

# if the file is xls
if input_path.endswith('.xls'):
    csv_file_path = xls2csv(input_path)
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    process_csv_data(csv_file_path, 100)

# if the file is csv
elif input_path.endswith('.csv'):
    csv_file_path = input_path
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    process_csv_data(csv_file_path, 100)


else:
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
        csv_path=gray_level_detct(input_path, 20, 40, 0, 0)
        base_name =os.path.splitext(os.path.basename(csv_path))[0]
        process_csv_data(csv_path, 240)

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

    choice = input("Do u wanna extract the modal_information of the pre_video? (y/n)")
    if choice == "y":
        
        # edge_choice = input("Press '1' to use darkest_point algorithm or '2' to use canny edge_detect algorthm: ").strip().upper()

        # if edge_choice == "1":
        #     frequencies = frequency(input_path, x, y, w, h, 'darkest')
        # elif edge_choice == "2":
        #     frequencies = frequency(input_path, x, y, w, h, 'canny')

        # first_frequency = frequencies[0]
        # print("Selected Frequencies:", frequencies, "Hz")
        # print("First_frequency = ", first_frequency)

        frequencies_darkest = frequency(input_path, x, y, w, h, 'darkest')
        frequencies_canny = frequency(input_path, x, y, w, h, 'canny')

        # print("Selected Frequencies(darkest):", frequencies_darkest, "Hz")
        # print("Selected Frequencies(canny):", frequencies_canny, "Hz")
        # print("Selected Frequencies(darkest):", frequencies_darkest, "Hz")
        # print(("Selected Frequencies(canny):", frequencies_canny, "Hz"))

    # 提取振型 TODO
    # choice = input("Do u wanna extract vibration_mode? (y/n)").strip().upper()
    # if choice == "y":
    #     first_frequency = frequencies[0]
    #     y_matrixs, start_col, end_col = video_edge(input_path)
    #     amplitude_by_col = model(y_matrixs, start_col, end_col, 60, first_frequency)

    # 设置视频源、输出文件名、最大帧数等参数
    vidFname = input_path
    maxFrames = 20000
    windowSize = 30
    # factor = 10
    fpsForBandPass = 90
    # lowFreq = 4
    # highFreq = 8

    input_values = input("Enter low frequency, high frequency, and factor separated by commas (e.g., 4,8,10): ")
    lowFreq, highFreq, factor = map(int, input_values.split(','))

    # 构建输出文件名
    vidFnameOut = os.path.join('media_mag', os.path.basename(vidFname).replace('.mp4', '-Mag%d_Ideal-lo%d-hi%d.avi' % (factor, lowFreq, highFreq)))
    
    current_directory = os.getcwd()
    video_name = os.path.splitext(os.path.basename(vidFname))[0]
    media_directory = os.path.join(current_directory, "media/media_mag")
    input_video_directory = os.path.join(media_directory, video_name)
    check_path = os.path.join(input_video_directory, os.path.basename(vidFname).replace('.mp4', '_roi(%d,%d,%d,%d)_Mag%d_Ideal-lo%d-hi%d.avi' % (x, y, w, h, factor, lowFreq, highFreq)))

    # # 检查是否放大过（同样参数）
    # if os.path.exists(check_path):
    #     print(f"The file {check_path} exists.")
    #     mag_vid_path = check_path
    # else:
    #     print(f"The file {check_path} does not exist.")   
        # 开始运动放大
    mag_vid_path = phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq, x, y, w, h)
        # 可选：将输出视频转换为MP4格式
        # print("\nStart convert video to mp4\n")
        # subprocess.call(['ffmpeg', '-i', mag_vid_path, mag_vid_path.replace('.avi', '.mp4')])

    # edge_choice = input("Press '1' to use darkest_point algorithm or '2' to use canny edge_detect algorthm: ").strip().upper()
    # if edge_choice == "1":
    #     frequencies = frequency(mag_vid_path, x, y, w, h, 'darkest after amplified')
    # elif edge_choice == "2":
    #     frequencies = frequency(mag_vid_path, x, y, w, h, 'canny after amplified')

    frequencies = frequency(mag_vid_path, x, y, w, h, 'darkest after amplified')
    # frequencies = frequency(mag_vid_path, x, y, w, h, 'canny after amplified')