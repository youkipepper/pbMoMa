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

rotate_choice = False
extract_pre = True
input_path = input("Enter the path of the file: ")

# if the file is xls
if input_path.endswith('.xls'):
    csv_file_path = xls2csv(input_path)
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    process_csv_data(csv_file_path, 100, mark_point= True) # notation 设置xls采样率

# if the file is csv
elif input_path.endswith('.csv'):
    csv_file_path = input_path
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    process_csv_data(csv_file_path, 100, mark_point= True) # notation 设置csv采样率

else:
    if rotate_choice == True:
        cap = cv2.VideoCapture(input_path)
        points = select_two_points(cap)
        if points:
            angle = calculate_rotation_angle(points)
            input_path = rotate_video(input_path, angle)  # 更新input_path为旋转后的视频路径

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

    # choice = input("Do u wanna extract the modal_information of the pre_video? (y/n)")
    # if choice == "y":
    if extract_pre == True:
        # frequencies_canny = frequency(input_path, x, y, w, h, 'canny', save_csv= True, show_video= True, mark_point= False, save_video= False, lable= 'cable') # notation 放大前 canny
        frequencies_darkest = frequency(input_path, x, y, w, h, 'darkest', save_csv= True, show_video= True, mark_point= True, save_video= False, lable= 'cable') # notation 放大前 darkest
        frequencies_darkest = frequency(input_path, x, y, w, h, 'dark_gray', save_csv= True, show_video= True, mark_point= True, save_video= False, lable= 'cable') # notation 放大前 darkest
        # frequencies_gray = frequency(input_path, x, y, w, h, 'gray', save_csv= True, show_video= True, mark_point= False, save_video= False, lable= 'cable') # notation 放大前 dark_gray

        # frequencies_canny = frequency(input_path, x, y, w, h, 'canny', save_csv= True, show_video= True, mark_point= True, save_video= False, lable= 'road') # notation 放大前 canny
        # frequencies_gray = frequency(input_path, x, y, w, h, 'gray', save_csv= True, show_video= True, mark_point= True, save_video= False, lable= 'road') # notation 放大前 dark_gray

        # print("Selected Frequencies(darkest):", frequencies_darkest, "Hz")
        # print("Selected Frequencies(canny):", frequencies_canny, "Hz")
        # print("Selected Frequencies(darkest):", frequencies_darkest, "Hz")
        # print(("Selected Frequencies(canny):", frequencies_canny, "Hz"))

    # todo 提取振型 
    # choice = input("Do u wanna extract vibration_mode? (y/n)").strip().upper()
    # if choice == "y":
    #     first_frequency = frequencies[0]
    #     y_matrixs, start_col, end_col = video_edge(input_path)
    #     amplitude_by_col = model(y_matrixs, start_col, end_col, 60, first_frequency)

    # 设置视频源、输出文件名、最大帧数等参数
    vidFname = input_path
    maxFrames = 20000
    windowSize = 30
    fpsForBandPass = 90
    # factor = 10
    # lowFreq = 4
    # highFreq = 8

    input_values = input("Enter low frequency, high frequency, and factor separated by commas (e.g., 4,8,10): ")
    lowFreq, highFreq, factor = map(int, input_values.split(','))
    
    current_directory = os.getcwd()
    video_name = os.path.splitext(os.path.basename(vidFname))[0]
    media_directory = os.path.join(current_directory, "media/media_mag")
    input_video_directory = os.path.join(media_directory, video_name)
    check_path = os.path.join(input_video_directory, os.path.basename(vidFname).replace('.mp4', '_roi(%d,%d,%d,%d)_Mag%d_Ideal-lo%d-hi%d.avi' % (x, y, w, h, factor, lowFreq, highFreq)))
    mag_vid_path = phaseBasedMagnify(vidFname, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq, x, y, w, h)

    # avi2mp4
    # print("\nStart convert video to mp4\n")
    # subprocess.call(['ffmpeg', '-i', mag_vid_path, mag_vid_path.replace('.avi', '.mp4')])

    # frequencies = frequency(mag_vid_path, x, y, w, h, 'canny_amplified', save_csv= True, show_video= True, mark_point= False, save_video= False) #notation 放大后 canny
    frequencies = frequency(mag_vid_path, x, y, w, h, 'darkest_amplified', save_csv= True, show_video= True, mark_point= True, save_video= False) # notation 放大后 darkest
    frequencies = frequency(mag_vid_path, x, y, w, h, 'dark_gray_amplified', save_csv= True, show_video= True, mark_point= True, save_video= False) #notation 放大后 dark_gray
    # frequencies = frequency(mag_vid_path, x, y, w, h, 'gray_amplified', save_csv= True, show_video= True, mark_point= False, save_video= False) #notation 放大后 gray
