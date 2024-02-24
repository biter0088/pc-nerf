import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import struct
import sys
import open3d as o3d
import os
import math
import pcl #add hxz

from tqdm import tqdm
import time

"""
    1. 修改自 eval_multi_frame_maicity_view_0406.py 的 multi_frame_maicity_test_sub_nerf()
"""
def pose2poselidar(pose_path=None, pose_lidar_path=None):
    ###2. 获取每帧点云的位姿和位置信息
    # 加载每帧的位姿数据,并转换为激光雷达相对于全局坐标系的位姿
    T_velo2cam=np.array([[4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02],
                                    [-7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02],
                                    [9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01],
                                    [0,0,0,1]])#拓展为4*4矩阵    
    poses=[]
    file = open(pose_path, "r", encoding="utf-8")
    rows = file.readlines()# 读取文件中的所有行数据
    new_list = [row.strip() for row in rows]# 读取文件中的所有行数据
    for row in new_list:# 遍历文件中的所有行数据
        P_tmp=np.array([[0,0,0,1]])
        P_array = np.append(np.array([float(i) for i in row.strip('\n').split(' ')]).reshape(3,4), P_tmp, axis=0)
        P_array= np.matmul(P_array, T_velo2cam)# 转换为激光雷达相对于全局坐标系的位姿        
        poses.append(P_array)    
    file.close()
    poses=np.array(poses)

    # 求 data_start 时刻矩阵的逆矩阵
    T_start=poses[0]     # 修改  
    T_start_inv = np.linalg.inv(T_start)
    T_start_inv = torch.from_numpy(T_start_inv).float()        
    poses = torch.Tensor(poses)    
    poses_start= T_start_inv @ poses # 将全局坐标系下的 pose 转换到 起始帧坐标系下
    # print("poses_start:",poses_start)
    print("poses_start.shape:",poses_start.shape)    
    print("poses_start[0]:",poses_start[0])
    poses_start_select = poses_start[:,:3,:]
    poses_start_select=poses_start_select.numpy()
    print("poses_start_select.shape:",poses_start_select.shape)    
    print("poses_start_select[0]:",poses_start_select[0])
    
    if 1:
        with open(pose_lidar_path, 'w') as f:
            for matrix in poses_start_select:
                flat_matrix = matrix.flatten()
                f.write(' '.join(map(str, flat_matrix)) + '\n')



if __name__ == '__main__':

    pose_path= "/home/biter/paper2/kitti/10_base/poses.txt"
    pose_lidar_path= "/home/biter/paper2/kitti/10_base/poses_lidar.txt"    
    pose2poselidar(pose_path=pose_path, pose_lidar_path=pose_lidar_path)
    