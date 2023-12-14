from perceptual.filterbank import *
import cv2
import sys
import numpy as np
from pyr2arr import Pyramid2arr
from temporal_filters import IdealFilterWindowed, ButterBandpassFilter
import subprocess
import os

from progress_bar import print_progress_bar

# 用户选择ROI模式
def select_roi_mode(vidReader, width, height):
    print("Select ROI mode: 1) Full video 2) Manual selection 3) Enter ROI")
    choice = input("Enter choice (1, 2, or 3): ").strip()

    if choice == '1':
        # 使用整个视频
        return 0, 0, width, height
    elif choice == '2':
        # 手动选择ROI
        _, first_frame = vidReader.read()
        roi = cv2.selectROI("Select ROI", first_frame, fromCenter=False)
        cv2.destroyWindow("Select ROI")
        return roi
    elif choice == '3':
        # 用户输入ROI参数
        while True:
            try:
                x, y = map(int, input("Enter ROI top-left corner (x, y): ").split(','))
                w, h = map(int, input("Enter ROI size (width, height): ").split(','))
                # 确保ROI在图像范围内
                if 0 <= x < width and 0 <= y < height and x + w <= width and y + h <= height:
                    print(f"Top-left corner: ({x},{y}), ROI Size: {w}x{h}")
                    return x, y, w, h
                else:
                    print("ROI is out of image bounds. Please try again.")
            except ValueError:
                print("Invalid input. Please enter integers.")

# 基于相位的视频放大函数
def phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq, x, y, w, h):
    # 初始化可导向复数金字塔
    steer = Steerable(5)
    pyArr = Pyramid2arr(steer)

    # 读取视频文件
    print("Reading:", vidFname)

    # 获取视频属性
    vidReader = cv2.VideoCapture(vidFname)

    # 获取原视频的 FourCC 编码
    original_fourcc = int(vidReader.get(cv2.CAP_PROP_FOURCC))
    fourcc_chars = "".join([chr((original_fourcc >> 8 * i) & 0xFF) for i in range(4)])

    vidFrames = int(vidReader.get(cv2.CAP_PROP_FRAME_COUNT))    
    width = int(vidReader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidReader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vidReader.get(cv2.CAP_PROP_FPS))
    func_fourcc = cv2.VideoWriter_fourcc

    # 如果fps未知，则默认为30
    if np.isnan(fps):
        fps = 30

    # 输出视频的帧数、分辨率和帧率
    print(' %d frames' % vidFrames)
    print(' (%d x %d)' % (width, height))
    print(' FPS:%d' % fps)

    # # 选择ROI
    # _, first_frame = vidReader.read()
    # roi = cv2.selectROI("Select ROI", first_frame, fromCenter=False)
    # x, y, w, h = map(int, roi)
    # roi_center = (x + w // 2, y + h // 2)
    # print(f"ROI Center: {roi_center}, ROI Size: {w}x{h}")
    # cv2.destroyWindow("Select ROI")

    print(f"ROI Top-Left Corner: ({x}, {y}), ROI Size: {w}x{h}")

    # 获取原视频的文件名（不包括扩展名）
    base_name = os.path.splitext(os.path.basename(vidFname))[0]

    # 构建输出目录路径（在media_mag文件夹中，以原视频文件名命名的子文件夹）
    output_dir = os.path.join(os.path.dirname(os.path.dirname(vidFname)), "media_mag", base_name)

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 构建输出视频的新路径（包含ROI信息）
    output_vid_path = os.path.join(output_dir, f"{base_name}_roi({x},{y},{w},{h})_Mag{factor}_Ideal-lo{lowFreq}-hi{highFreq}.avi")

    # 视频写入器设置
    fourcc = cv2.VideoWriter_fourcc(*fourcc_chars)
    vidWriter = cv2.VideoWriter(output_vid_path, fourcc, int(fps), (width, height), 1)
    print('Writing:', vidFnameOut)

    # 处理的帧数
    nrFrames = min(vidFrames, maxFrames)

    # 设置时域滤波器
    filter = IdealFilterWindowed(windowSize, lowFreq, highFreq, fps=fpsForBandPass, outfun=lambda x: x[0])

    # 逐帧读取并处理视频
    print('FrameNr:')
    for frameNr in range(nrFrames + windowSize):
        print_progress_bar(frameNr, nrFrames, prefix='Amplifying motion:', suffix='Complete')
        # print(f"Processing Frame {frameNr} / {nrFrames}")
        sys.stdout.flush()  # 刷新输出

        if frameNr < nrFrames:
            # 读取一帧
            _, im = vidReader.read()
            if im is None:
                break

            # 转换为灰度图像
            if len(im.shape) > 2:
                grayIm = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            else:
                grayIm = im

            # 只处理ROI区域
            grayIm_roi = grayIm[y:y+h, x:x+w]
            
            # ###rgb_p1
            # # 从彩色图像中提取亮度通道
            # im_roi_color = im[y:y+h, x:x+w]
            # grayIm_roi = cv2.cvtColor(im_roi_color, cv2.COLOR_RGB2GRAY)
            # ###rgb_p1

            # 构建金字塔并获取系数
            coeff = steer.buildSCFpyr(grayIm_roi)

            # 添加到视频数组
            arr = pyArr.p2a(coeff)
            phases = np.angle(arr)

            # 更新时域滤波器
            filter.update([phases])

            # 尝试获取滤波输出
            try:
                filteredPhases = filter.next()
            except StopIteration:
                continue

            # 运动放大
            magnifiedPhases = (phases - filteredPhases) + filteredPhases * factor

            # 重建新数组
            newArr = np.abs(arr) * np.exp(magnifiedPhases * 1j)
            newCoeff = pyArr.a2p(newArr)

            # 重建金字塔并获取输出图像
            out = steer.reconSCFpyr(newCoeff)

            # 转换为RGB图像
            rgbIm = np.empty((out.shape[0], out.shape[1], 3))
            rgbIm[:, :, 0] = out
            rgbIm[:, :, 1] = out
            rgbIm[:, :, 2] = out

            # 写入到磁盘
            res = cv2.convertScaleAbs(rgbIm)
            vidWriter.write(res)

            # 将ROI结果放回原始图像
            res = cv2.convertScaleAbs(rgbIm)
            im[y:y+h, x:x+w] = res

            # ###rgb_p2
            # # 将输出转换为uint8类型
            # out_uint8 = cv2.convertScaleAbs(out)

            # # 创建mask，确定哪些像素发生了变化
            # mask = cv2.absdiff(grayIm_roi, out_uint8) > 0
            # mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

            # # 将处理后的亮度通道与原始彩色图像的颜色通道合并
            # out_rgb = cv2.cvtColor(out_uint8, cv2.COLOR_GRAY2RGB)
            # im_roi_color = np.where((mask_3d), out_rgb, im_roi_color)

            # # 将处理后的ROI区域放回原始图像
            # im[y:y+h, x:x+w] = im_roi_color
            # ###pgb_p2

            vidWriter.write(im)

    # 释放视频读写器资源
    vidReader.release()
    vidWriter.release()

    # 返回输出视频的路径
    return output_vid_path

# 主脚本部分
if __name__ == "__main__":
    # 设置视频源、输出文件名、最大帧数等参数
    vidFname = '/Users/youkipepper/Desktop/pbMoMa/media/2023-11-30/video_231130_02.mp4'
    vidReader = cv2.VideoCapture(vidFname)
    width = int(vidReader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidReader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 用户选择ROI模式
    x, y, w, h = select_roi_mode(vidReader, width, height)

    maxFrames = 60000
    windowSize = 30
    factor = 5
    fpsForBandPass = 600
    lowFreq = 4
    highFreq = 15
    # vidFnameOut = vidFname + '-Mag%dIdeal-lo%d-hi%d.avi' % (factor, lowFreq, highFreq)

    # 构建输出文件名
    vidFnameOut = os.path.join('media_mag', os.path.basename(vidFname).replace('.mp4', '-Mag%dIdeal-lo%d-hi%d.avi' % (factor, lowFreq, highFreq)))

    # 可选：将输出视频转换为MP4格式
    output_vid_path = phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq, x, y, w, h)
    subprocess.call(['ffmpeg', '-i', output_vid_path, output_vid_path.replace('.avi', '.mp4')])