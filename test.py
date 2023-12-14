import subprocess

xls_files = [
    '/Users/youkipepper/Desktop/pbMoMa/data/cc231203/231203-1/231203-1.xls',
    '/Users/youkipepper/Desktop/pbMoMa/data/cc231203/231203-2/231203-2.xls',
    '/Users/youkipepper/Desktop/pbMoMa/data/cc231203/231203-3/231203-3.xls',
    '/Users/youkipepper/Desktop/pbMoMa/data/cc231203/231203-4/231203-4.xls',
    '/Users/youkipepper/Desktop/pbMoMa/data/cc231203/231203-5/231203-5.xls',
    '/Users/youkipepper/Desktop/pbMoMa/data/cc231203/231203-6/231203-6.xls'
]

for xls_file in xls_files:
    preset_inputs = [
        xls_file,  # 输入视频路径
        'y' # 不标记

    ]

    # 启动进程并通过stdout和stdin将预设输入传递给代码，并捕获输出
    process = subprocess.Popen(['python', 'main.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 将预设输入发送到进程
    for user_input in preset_inputs:
        process.stdin.write(user_input + '\n')

    # 关闭stdin
    process.stdin.close()

    # 读取并打印程序的输出
    while process.poll() is None:
        output = process.stdout.readline()
        if output:
            print(output.strip())


video_files = [
    '/Users/youkipepper/Desktop/pbMoMa/media/2023-12-3/video_231203_01.MP4',
    '/Users/youkipepper/Desktop/pbMoMa/media/2023-12-3/video_231203_02.MP4',
    '/Users/youkipepper/Desktop/pbMoMa/media/2023-12-3/video_231203_03.MP4',
    '/Users/youkipepper/Desktop/pbMoMa/media/2023-12-3/video_231203_04.MP4',
    '/Users/youkipepper/Desktop/pbMoMa/media/2023-12-3/video_231203_05.MP4',
    '/Users/youkipepper/Desktop/pbMoMa/media/2023-12-3/video_231203_06.MP4'
]

for video_file in video_files:
    preset_inputs = [
        video_file,  # 输入视频路径
        'n',  # 旋转视频
        'n',  # 灰度检测
        'E',  # 输入roi信息
        '1505,312', # roi脚点
        '73,180', # roi尺寸
        'y', # 提取放大前
        ### 提取信息
        # 'n', # 不标记
        # 'n', # 不标记
        'n', # 不标记
        'n', # 不标记

    ]

    # 启动进程并通过stdout和stdin将预设输入传递给代码，并捕获输出
    process = subprocess.Popen(['python', 'main.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 将预设输入发送到进程
    for user_input in preset_inputs:
        process.stdin.write(user_input + '\n')

    # 关闭stdin
    process.stdin.close()

    # 读取并打印程序的输出
    while process.poll() is None:
        output = process.stdout.readline()
        if output:
            print(output.strip())
