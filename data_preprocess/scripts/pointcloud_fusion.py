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
def multi_frame_pointcloud_fusion(root_dir, data_start=1,data_end=2,
                            range_delete_x=2, range_delete_y=1, range_delete_z=0.5,
                            interest_x=12, interest_y=6,                              
                            pose_path=None,
                            save_path=None,
                            over_height=0.168,over_low=-2):

    T_velo2cam=np.array([[4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02],
                                    [-7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02],
                                    [9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01],
                                    [0,0,0,1]])# 4*4    
    poses=[]
    file = open(pose_path, "r", encoding="utf-8")
    rows = file.readlines()
    new_list = [row.strip() for row in rows]
    for row in new_list:
        P_tmp=np.array([[0,0,0,1]])
        P_array = np.append(np.array([float(i) for i in row.strip('\n').split(' ')]).reshape(3,4), P_tmp, axis=0)
        P_array= np.matmul(P_array, T_velo2cam)        
        poses.append(P_array)    
    file.close()
    poses=np.array(poses)
    positions=poses[: ,:3,-1]
    positions=positions

    T_start=poses[data_start+1]    
    T_start_inv = np.linalg.inv(T_start)
    T_start_inv = torch.from_numpy(T_start_inv).float()        
    poses = torch.Tensor(poses)    
    poses_start= T_start_inv @ poses 

    source_save_all=np.ones((0,3))
    pose_save_all =np.ones((0,3))    
    count_train = 0
    for j in range(data_start, data_end):    
        if((j+1-3)%5!=0): # test set: hold one form every  5 scans
            count_train = count_train + 1
        else:
            continue                  
     
        file_path = os.path.join(root_dir, '{}.pcd'.format(j+1))                     
        print("file_path: ",file_path)           
        points_source = pcl.PointCloud()    
        points_source = pcl.load(file_path)
        print("count: ",points_source.size)
        
        count_effective=0       
        points_effective_tmp=np.zeros((4, points_source.size),dtype=float)                             
        for i in range(0, points_source.size):     
            if(abs(points_source[i][0])<range_delete_x and abs(points_source[i][1])<range_delete_y and abs(points_source[i][2])<range_delete_z):                                     
                continue   
            # over high
            if(points_source[i][2]>over_height) or (points_source[i][2]<over_low):
                continue
            # over distant
            range_tmp=np.sqrt(np.square(points_source[i][0])+np.square(points_source[i][1])+np.square(points_source[i][2]))
            if(range_tmp>120):
                continue        
              
            points_effective_tmp[0][count_effective]=points_source[i][0]                      
            points_effective_tmp[1][count_effective]=points_source[i][1]         
            points_effective_tmp[2][count_effective]=points_source[i][2]          
            points_effective_tmp[3][count_effective]=1       
            count_effective=count_effective+1
        print("count: ",count_effective)                
        points_effective=points_effective_tmp[:, :count_effective]              
        
        # global frame
        points_effective = poses_start[j+1] @ points_effective                 
        points_effective = points_effective.T[:,:3]              
        
        count_nerf=0 # in nerf
        points_nerf_tmp=np.zeros((points_effective.shape[0], 3),dtype=float)                      
        for i in range(0, points_effective.shape[0]):        
            near_all_pose=False 
            for k in range(data_start, data_end):
                if(abs(points_effective[i][0]-poses_start[k+1][0,-1])>interest_x or abs(points_effective[i][1]-poses_start[k+1][1,-1])>interest_y ):     # 修改 0617                                             
                    continue
                else:
                    near_all_pose=True
                    break
            if(near_all_pose==False):
                continue
                
            points_nerf_tmp[count_nerf][0]=points_effective[i][0]                     
            points_nerf_tmp[count_nerf][1]=points_effective[i][1]         
            points_nerf_tmp[count_nerf][2]=points_effective[i][2]        
            
            count_nerf=count_nerf+1
        print("count: ",count_nerf)                
        points_nerf=points_nerf_tmp[:count_nerf]        #  0415 -2          

        if(1):
            source_save=np.zeros((points_nerf.shape[0],3),dtype=float)
            for i in range(0, points_nerf.shape[0]):          
                source_save[i][0]=points_nerf[i][0]
                source_save[i][1]=points_nerf[i][1]         
                source_save[i][2]=points_nerf[i][2]         
            if(j==data_start):
                source_save_all=source_save
            else:                   
                source_save_all   = np.concatenate([source_save_all, source_save],axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(source_save_all)                
            file_path = os.path.join(save_path, 'source.pcd')              
            o3d.io.write_point_cloud(file_path, pcd)

        if(1):
            pose_save=np.ones((4,1),dtype=float)
            pose_save[0][0]=poses[j+1][0,-1]
            pose_save[1][0]=poses[j+1][1,-1]       
            pose_save[2][0]=poses[j+1][2,-1]                            
            pose_save=T_start_inv @ pose_save
            pose_save = pose_save.T[:,:3]                     
            
            pose_save_all   = np.concatenate([pose_save_all, pose_save],axis=0)
            pcd_pose = o3d.geometry.PointCloud()
            pcd_pose.points = o3d.utility.Vector3dVector(pose_save_all)                
            file_path = os.path.join(save_path, 'pose.pcd')              
            o3d.io.write_point_cloud(file_path, pcd_pose)

    print("train set number:",count_train)
    source_pointcloud_path =  os.path.join(save_path, 'source.pcd')
    pcd = o3d.io.read_point_cloud(source_pointcloud_path)
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    print("nerf in width: [",min_bound[1],", ",max_bound[1],"]")         
    print("nerf in length: [",min_bound[0],", ",max_bound[0],"]")                 
    print("nerf in height: [",min_bound[2],", ",max_bound[2],"]")           

if __name__ == '__main__':

    Lidar_height=0.168 
    data_start=1150    # scan order number from    data_start+1
    data_end=1200            
    root_dir="data_preprocess/kitti/dataset/sequences/00/pcd_remove_dynamic1151_1200"
    kwargs = {'root_dir': root_dir, 'data_start':data_start, 'data_end':data_end, 
                        'range_delete_x':3, 'range_delete_y':2, 'range_delete_z':1.25,
                        'interest_x':20,'interest_y':20, # interest area of single scan 
                        'pose_path':"data_preprocess/kitti/dataset/sequences/00/poses.txt",
                        'save_path':"data_preprocess/kitti_pre_processed/sequence00/1151_1200_view/",
                        'over_height':Lidar_height, 'over_low':-2
                        }   
    
    multi_frame_pointcloud_fusion(**kwargs)
    