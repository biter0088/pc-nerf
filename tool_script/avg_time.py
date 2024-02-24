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


if __name__ == '__main__':
    use_ray_count = 1
    data_start = 750  # 选取点云的序号从: data_start+1 ~ data_end
    data_end =  800 # 下一个十字路口前   
    count_train = 0
    sum = 0
    for j in range(data_start, data_end):    # 配置文件中 j 从0开始        

        # if((j+1-3-data_start)%5==0):          # 1/5 用于测试
        # if((j+1-data_start)%4==0):                 # 1/4         
        # if((j+1-data_start)%3==0):                 # 1/3                      
        # if((j+1-data_start)%2==0):                 # 1/2
        # if((j+1-1-data_start)%3!=0):                 # 2/3
        # if((j+1-3-data_start)%5!=0):                 # 4/5
        if((j+1-5-data_start)%10!=0):                 # 9/10
            count_train = count_train + 1
            select_id=j+1
        else:
            continue  
        print("select_id:",select_id)
        if count_train==1:
        # if select_id==1003:
            print("select_id:",select_id)        
            if  use_ray_count==0: # 获取点云的数目
                pcd_0 = o3d.io.read_point_cloud("/home/biter/paper2/sequence03_big25/751_800_101/source_"+str(select_id)+".pcd")    
                print("/home/biter/paper2/sequence03_big25/751_800_101/source_"+str(select_id)+".pcd")
                pcd_0_np = np.asarray(pcd_0.points)            
                num_0 = pcd_0_np.shape[0]        
            if use_ray_count: # 获取相交射线的数目
                ray_0 = np.load("/home/biter/paper2/pc_nerf/logs/sequence03_big25/751_800_101/渲染3/"+str(select_id)+"pcd/子场2_3/all_ranges_child.npy")
                print("/home/biter/paper2/pc_nerf/logs/sequence03_big25/751_800_101/渲染3/"+str(select_id)+"pcd/子场2_3/all_ranges_child.npy")
                num_0 = ray_0.shape[0]        

        if use_ray_count==0:
            pcd_cur = o3d.io.read_point_cloud("/home/biter/paper2/sequence03_big25/751_800_101/source_"+str(select_id)+".pcd")    
            pcd_cur_np = np.asarray(pcd_cur.points)            
            sum = sum + pcd_cur_np.shape[0]        
        if use_ray_count:
            ray_cur = np.load("/home/biter/paper2/pc_nerf/logs/sequence03_big25/751_800_101/渲染3/"+str(select_id)+"pcd/子场2_3/all_ranges_child.npy")        
            sum = sum + ray_cur.shape[0]        

    time_0 =12+3/60
    # time_0 = 9
    print("time_0:",time_0)
    print("count_train:",count_train)    
    print("sum:",sum)
    print("num_0:",num_0)    
    # print("平均时间:", time_0/num_0*sum/10)
    print("平均时间:", time_0/num_0*sum/count_train)    
    print("平均时间/min:", time_0/num_0*sum/count_train/60)        
   