import os


def get_folders(directory):
    folders = []
    # 遍历目录中的所有文件和文件夹
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        # 判断路径是否为文件夹
        if os.path.isdir(item_path):
            folders.append(item_path)
    return folders
