def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    打印进度条和百分比
    :param iteration: 当前迭代次数
    :param total: 总迭代次数
    :param prefix: 前缀字符串
    :param suffix: 后缀字符串
    :param length: 进度条长度
    :param fill: 进度条填充字符
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix} {iteration}/{total}', end="\r")
    if iteration == total: 
        print()
