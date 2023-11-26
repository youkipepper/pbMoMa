import os
from xls2csv import xls2csv
from process_data import process_csv_data

csv_file_path=xls2csv('/Users/youkipepper/Desktop/pbMoMa/data/cc231121/231121-01/231121-011-15.xls')
base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
process_csv_data(csv_file_path, 100, base_name)
