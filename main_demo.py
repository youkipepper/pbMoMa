import sys
import subprocess

# def read_file_paths(file_path):
#     with open(file_path, 'r') as file:
#         return [line.strip() for line in file]

# def read_file_paths(file_path):
#     encodings = ['utf-8', 'iso-8859-1', 'cp1252']  # 常见的几种编码
#     for encoding in encodings:
#         try:
#             with open(file_path, 'r', encoding=encoding) as file:
#                 return [line.strip() for line in file]
#         except UnicodeDecodeError:
#             continue
#     raise ValueError(f"Cannot decode file {file_path} with any of the known encodings.")

# def read_file_paths(file_path):
#     # print("Reading file path:", file_path)  # 调试打印
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             return [line.strip() for line in file]
#     except (IOError, UnicodeDecodeError):
#         # 如果文件无法作为文本文件打开，或者编码不是utf-8，则假定它是直接的路径
#         return [file_path]

def read_file_paths(file_path):
    if file_path.endswith('.txt'):  # 检查是否为文本文件
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return [line.strip() for line in file]
        except (IOError, UnicodeDecodeError):
            print(f"Error reading file: {file_path}")
            return []
    else:
        return [file_path]  # 如果不是文本文件，直接返回路径



def is_video_file(file_path):
    video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv']
    return any(file_path.lower().endswith(ext) for ext in video_extensions)


def process_file(file_path, file_type):
    if file_type == 'xls':
        preset_inputs_xls = [
            file_path,  # 输入路径
            'y' # mark
        ]
        execute_process(preset_inputs_xls)

    elif file_type == 'csv':

        # print("Processing CSV file:", file_path)  # 调试打印

        preset_inputs_csv = [
            file_path,  # 输入路径
            'n' # mark
        ]
        execute_process(preset_inputs_csv)        
    
    else:
        preset_inputs = [
            file_path,  # 输入视频路径
            'n',  # 旋转视频
            'n',  # 灰度检测
            'E',  # 输入roi信息
            '629,385', # roi脚点
            '40,137', # roi尺寸
            'y', # 提取放大前
            'n', # darkest mark
            'n', # canny mark 
            '5,13,10', # 放大参数
            'n', # darkest mark
        ]
        execute_process(preset_inputs)

        preset_inputs = [
            file_path,  # 输入视频路径
            'n',  # 旋转视频
            'n',  # 灰度检测
            'E',  # 输入roi信息
            '629,385', # roi脚点
            '40,137', # roi尺寸
            'n', # 提取放大前
            # 'n', # darkest mark
            # 'n', # canny mark 
            '5,13,15', # 放大参数
            'n', # darkest mark
        ]
        execute_process(preset_inputs)

        preset_inputs = [
            file_path,  # 输入视频路径
            'n',  # 旋转视频
            'n',  # 灰度检测
            'E',  # 输入roi信息
            '629,385', # roi脚点
            '40,137', # roi尺寸
            'n', # 提取放大前
            # 'n', # darkest mark
            # 'n', # canny mark 
            '13,22,10', # 放大参数
            'n', # darkest mark
        ]
        execute_process(preset_inputs)

        preset_inputs = [
            file_path,  # 输入视频路径
            'n',  # 旋转视频
            'n',  # 灰度检测
            'E',  # 输入roi信息
            '629,385', # roi脚点
            '40,137', # roi尺寸
            'n', # 提取放大前
            # 'n', # darkest mark
            # 'n', # canny mark 
            '13,22,15', # 放大参数
            'n', # darkest mark
        ]
        execute_process(preset_inputs)
        
           

def execute_process(preset_inputs):

    # print("Executing process with:", preset_inputs)  # 调试打印

    process = subprocess.Popen(['python', 'main.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for user_input in preset_inputs:
        process.stdin.write(user_input + '\n')
    process.stdin.close()

    while process.poll() is None:
        output = process.stdout.readline()
        if output:
            print("Output:", output.strip())

# def main():
#     file_paths = []
#     for file_path in sys.argv[1:]:
#         file_paths.extend(read_file_paths(file_path))
    
#     for path in file_paths:
#         if path.endswith('.xls'):
#             process_file(path, True)
#         elif is_video_file(path):
#             process_file(path, False)

def main():
    # print("Command line arguments:", sys.argv)  # 调试打印
    file_paths = []
    for file_path in sys.argv[1:]:
        file_paths.extend(read_file_paths(file_path))
        # print("Processed file paths:", file_paths)  # 新增的调试打印
    for path in file_paths:
        if path.endswith('.xls'):
            process_file(path, 'xls')
            # print("Processing as XLS")  # 新增的调试打印
        elif path.endswith('.csv'):
            process_file(path, 'csv')
            # print("Processing as CSV")  # 新增的调试打印
        elif is_video_file(path):
            # print("Processing as Video")  # 新增的调试打印
            process_file(path, 'video')

if __name__ == "__main__":
    main() 
