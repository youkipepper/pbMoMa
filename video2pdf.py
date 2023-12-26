import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from fpdf import FPDF
import os
from progress_bar import print_progress_bar

def video_to_pdf(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_pdf_path = os.path.join(os.path.dirname(video_path), f'{video_name}.pdf')

    cap = cv2.VideoCapture(video_path)

    # 获取视频的总帧数和帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Total Frames: {total_frames}")
    print(f"Frame Rate: {frame_rate} fps")
    print(f"Resolution: {frame_width} x {frame_height}")

    pdf = FPDF(unit="pt")  # 使用pt作为单位
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        print_progress_bar(frame_count, total_frames, prefix=f'Converting video to pdf :', suffix='Complete')

        height, width, _ = frame.shape

        # 将PDF页面大小设置为图像大小
        pdf.add_page()
        pdf.set_auto_page_break(0, margin=0)
        pdf.set_margins(0, 0, 0)
        pdf.set_fill_color(255, 255, 255)
        pdf.rect(0, 0, width, height, style='F')

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        fig = Figure(figsize=(width / 100, height / 100), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.imshow(frame)
        ax.axis('off')

        temp_img = f"temp_frame_{frame_count}.png"
        fig.savefig(temp_img, dpi=100)

        pdf.image(temp_img, x=0, y=0, w=width, h=height)

        os.remove(temp_img)

    cap.release()
    pdf.output(output_pdf_path, "F")

video_to_pdf("/Users/youkipepper/Desktop/pbMoMa/gauss_fit_video.mp4")
