import os

folder_path = '/home/biter/paper2/maicity_dataset/00/velodyne'  # 文件夹路径

file_list = os.listdir(folder_path)  # 获取文件夹中的文件列表
print("len(file_list):",len(file_list))
file_list.sort()  # 按照文件名排序

for i, file_name in enumerate(file_list, start=1):
    file_extension = os.path.splitext(file_name)[1]  # 获取文件名后缀
    new_file_name = f"{i}{file_extension}"  # 新的文件名
    
    old_file_path = os.path.join(folder_path, file_name)  # 原始文件路径
    new_file_path = os.path.join(folder_path, new_file_name)  # 新的文件路径
    
    os.rename(old_file_path, new_file_path)  # 执行文件重命名操作
