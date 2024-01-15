# notation func：xls to csv

import pandas as pd
import os

def xls2csv(xls_file_path):
    """
    将 XLS 文件的第二列从第二行开始的数据转换为 CSV 文件。
    :param xls_file_path: XLS 文件的路径
    :return: 生成的 CSV 文件的路径
    """
    # 读取 xls 文件
    data = pd.read_excel(xls_file_path, usecols=[1])  # 仅读取第二列，跳过第一行（标题）

    # 构造新的 DataFrame
    new_data = pd.DataFrame(data)

    # 定义 CSV 文件保存路径
    csv_output_folder = 'csv'
    os.makedirs(csv_output_folder, exist_ok=True)  # 如果文件夹不存在，则创建
    csv_file_name = os.path.basename(xls_file_path).replace('.xls', '.csv')
    csv_file_path = os.path.join(csv_output_folder, csv_file_name)

    # 将数据保存为 csv 文件
    new_data.to_csv(csv_file_path, index=False, header=False)  # 不包括索引和标题

    return csv_file_path

if __name__ == "__main__":
    # 测试函数
    xls2csv('./data/cc231121/231121-01/231121-011-15.xls')
