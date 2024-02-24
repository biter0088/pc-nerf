import os
import json
import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree
import torch
import torch.utils.data as data
import pcl #add hxz
import math
import time

subnerf_path=  "/home/biter/paper2/kitti_data/sequence03_big/751_800/sub_pointcloud/sub_nerf细化/split_child_nerf2/"
sub_nerf_test_num = 8957

sub_nerf_bound=np.zeros((sub_nerf_test_num,6)) # 依次存放: x_min, y_min, z_min, x_max, y_max, z_max
print("子场点云目录: ",subnerf_path)
for i in range(sub_nerf_test_num):    
    file_name = f"{i+1}.pcd"
    file_path = os.path.join(subnerf_path, file_name)
    pcd = o3d.io.read_point_cloud(file_path) # 读取点云数据
    bbox = pcd.get_axis_aligned_bounding_box() # 获得点云边界
    min_bound = bbox.get_min_bound() # 获得边界最小值和最大值
    max_bound = bbox.get_max_bound()
    extend_tmp0=0.0
    sub_nerf_bound[i][0]=min_bound[0]-extend_tmp0
    sub_nerf_bound[i][1]=min_bound[1]-extend_tmp0            
    sub_nerf_bound[i][2]=min_bound[2]-extend_tmp0
    sub_nerf_bound[i][3]=max_bound[0]+extend_tmp0           
    sub_nerf_bound[i][4]=max_bound[1]+extend_tmp0
    sub_nerf_bound[i][5]=max_bound[2]+extend_tmp0 

np.save("/home/biter/paper2/kitti_data/sequence03_big/751_800/sub_pointcloud/sub_nerf细化/"+"sub_nerf_bound",sub_nerf_bound) 
