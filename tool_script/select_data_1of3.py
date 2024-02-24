import os
import shutil

def copy_files_without_interval(source_folder, destination_folder, start_index=2, interval=5):


    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 获取源文件夹中按数值排序的文件列表
    files = sorted(os.listdir(source_folder))

    pcd_ornot = 0
    if 1:
        file_path = os.path.join(source_folder, files[0])
        if file_path.endswith(".pcd"):
            print(f"The file {file_path} has a '.pcd' extension.")
            pcd_ornot=1
        elif file_path.endswith(".bin"):
            print(f"The file {file_path} has a '.bin' extension.")
        else:
            print(f"The file {file_path} has an unknown extension.")

    # 遍历文件列表，从第三个文件开始，每隔5个不保留一个文件，并将剩余的文件复制到目标文件夹
    count = 0
    for i in range(len(files)):
        if (i - start_index) % interval != 0:
            source_file_path = os.path.join(source_folder, files[i])
            # destination_file_path = os.path.join(destination_folder, files[i]) # 保存为原来的序号
            if (pcd_ornot):
                destination_file_path = os.path.join(destination_folder, str(count)+".pcd") # 保存为新的的序号             
            else:
                destination_file_path = os.path.join(destination_folder, str(count)+".bin") # 保存为新的的序号                             
            count = count +1
            shutil.copy2(source_file_path, destination_file_path)
            print(f"文件 {files[i]} 已复制到 {destination_folder}")

def copy_lines_without_interval(source_file, destination_file, start_index=2, interval=5,train_less_test=0):
    # 确保目标文件夹存在
    destination_folder = os.path.dirname(destination_file)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 打开源文件和目标文件
    with open(source_file, 'r') as source, open(destination_file, 'w') as destination:
        # 读取源文件的所有行
        lines = source.readlines()

        # 遍历文件行，从第三行开始，每隔5行不保留一个，并将剩余的行复制到目标文件
        for i in range(len(lines)):
            if train_less_test==0: # 训练集比测试集合多            
                if (i - start_index) % interval != 0:
                    destination.write(lines[i])
            else:
                if (i - start_index) % interval == 0:
                    destination.write(lines[i])                


import os
import shutil

def copy_files_with_interval_and_rename(source_folder, destination_folder, start_index=2, interval=5,train_less_test=0):
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 获取源文件夹中按数值排序的文件列表
    files = sorted(os.listdir(source_folder))

    # 遍历文件列表，从第三个文件开始，每隔5个不保留一个文件，并将剩余的文件复制到目标文件夹
    count = 0
    for i in range(len(files)):
        if train_less_test==0: # 训练集比测试集合多
            if (i - start_index) % interval != 0:
                source_file_path = os.path.join(source_folder, files[i])
                # 解析文件名中的数字部分，生成新的文件名
                file_number = int(''.join(filter(str.isdigit, files[i])))
                # new_file_name = f"{(file_number - start_index) // interval:05d}.bin"
                new_file_name = f"{count:06d}.bin"          
                count = count + 1  
                destination_file_path = os.path.join(destination_folder, new_file_name)
                shutil.copy2(source_file_path, destination_file_path)
                print(f"文件 {files[i]} 已复制到 {destination_folder}，重命名为 {new_file_name}")
        else: # 训练集比测试集少
            if (i - start_index) % interval == 0:
                source_file_path = os.path.join(source_folder, files[i])
                # 解析文件名中的数字部分，生成新的文件名
                file_number = int(''.join(filter(str.isdigit, files[i])))
                # new_file_name = f"{(file_number - start_index) // interval:05d}.bin"
                new_file_name = f"{count:06d}.bin"          
                count = count + 1  
                destination_file_path = os.path.join(destination_folder, new_file_name)
                shutil.copy2(source_file_path, destination_file_path)
                print(f"文件 {files[i]} 已复制到 {destination_folder}，重命名为 {new_file_name}")            

start_index=0
interval = 3
train_less_test =1

source_folder_path = "/home/biter/paper2/kitti/10_base/velodyne"
destination_folder_path = "/home/biter/paper2/kitti/10_1of3/velodyne"
source_file_path = "/home/biter/paper2/kitti/10_base/poses_lidar.txt"
destination_file_path = "/home/biter/paper2/kitti/10_1of3/poses_lidar.txt"

# 例子：从a文件夹中按数值排序的文件中，从第三个文件开始，每隔5个不保留一个，将剩余的文件复制到b文件夹中
copy_files_with_interval_and_rename(source_folder_path, destination_folder_path, start_index=start_index, interval=interval,train_less_test=train_less_test)

# 例子：从a.txt文件中，从第三行开始，每隔5行不保留一个，将剩余的行复制到b.txt文件中
copy_lines_without_interval(source_file_path, destination_file_path, start_index=start_index, interval=interval,train_less_test=train_less_test)