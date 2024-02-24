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
                            interest_x=12, interest_y=6, # 针对单帧激光雷达设置感兴趣区域范围                             
                            pose_path="/home/biter/paper2/maicity_dataset/01/poses.txt",
                            save_path="/home/meng/subject/maicity数据预处理/maicity数据配置0511/",
                            over_height=0.168,over_low=-2,
                            select_id=0):

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
    positions=poses[: ,:3,-1]
    positions=positions

    # 求 data_start 时刻矩阵的逆矩阵
    # T_start=poses[data_start]
    T_start=poses[data_start+1]     # 修改 0617  
    T_start_inv = np.linalg.inv(T_start)
    T_start_inv = torch.from_numpy(T_start_inv).float()        
    poses = torch.Tensor(poses)    
    poses_start= T_start_inv @ poses # 将全局坐标系下的 pose 转换到 起始帧坐标系下

    # 3. 神经辐射场空间感兴趣区域内有效数据采样
    source_save_all=np.ones((0,3))
    pose_save_all =np.ones((0,3))    
    count_train = 0
    for j in range(data_start, data_end):    # 配置文件中 j 从0开始
        # if((j+1-3)%5!=0): # 测试集每5帧选取一帧，相应训练集每5帧选取4帧，第一帧从3.pcd开始
        # # if((j+1)%5!=0): # 测试集每5帧选取一帧，相应训练集每5帧选取4帧， 
        #     count_train = count_train + 1
        # else:
        #     continue                  


        if((j+1)==select_id): # 测试集每5帧选取一帧，相应训练集每5帧选取4帧，第一帧从3.pcd开始
        # if((j+1)==753): # 测试集每5帧选取一帧，相应训练集每5帧选取4帧，第一帧从3.pcd开始
            count_train = count_train + 1
        else:
            continue              
        
        file_path = os.path.join(root_dir, '{}.pcd'.format(j+1))                     
        print("点云 file_path: ",file_path)           
        points_source = pcl.PointCloud()    
        points_source = pcl.load(file_path)#这里的激光雷达点云数据应该是以激光雷达为坐标系的
        print("原始点云数量: ",points_source.size)
        
        # 判断是否是有效点云
        count_effective=0       
        points_effective_tmp=np.zeros((4, points_source.size),dtype=float)                             
        for i in range(0, points_source.size):     
            # 在激光雷达坐标系中进行判断，去除车体内部的点云    
            if(abs(points_source[i][0])<range_delete_x and abs(points_source[i][1])<range_delete_y and abs(points_source[i][2])<range_delete_z):                                     
                continue   
            # 去除过高的点云
            if(points_source[i][2]>over_height) or (points_source[i][2]<over_low):
                continue
            # 去除过远的点云
            range_tmp=np.sqrt(np.square(points_source[i][0])+np.square(points_source[i][1])+np.square(points_source[i][2]))
            if(range_tmp>120):
                continue        
              
            points_effective_tmp[0][count_effective]=points_source[i][0]                      
            points_effective_tmp[1][count_effective]=points_source[i][1]         
            points_effective_tmp[2][count_effective]=points_source[i][2]          
            points_effective_tmp[3][count_effective]=1#补全       
            count_effective=count_effective+1
        print("有效点云数量: ",count_effective)                
        points_effective=points_effective_tmp[:, :count_effective]              
        
        # 将点云转换到全局坐标系中
        points_effective = poses_start[j+1] @ points_effective    # 修改 0617                        
        points_effective = points_effective.T[:,:3]              
        
        count_nerf=0 # 属于神经辐射场的点云计数
        points_nerf_tmp=np.zeros((points_effective.shape[0], 3),dtype=float)                      
        for i in range(0, points_effective.shape[0]):        
            # 选取在车辆移动路径附近的点云，过远的点云舍弃
            near_all_pose=False # 初始化不靠近车辆行驶路径
            for k in range(data_start, data_end):
                # if(abs(points_effective[i][0]-poses_start[k][0,-1])>interest_x or abs(points_effective[i][1]-poses_start[k][1,-1])>interest_y ):               
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
        print("在神经辐射场内的点云数量: ",count_nerf)                
        # points_nerf=points_nerf_tmp[:, :count_nerf]        #  0415 -2                        
        points_nerf=points_nerf_tmp[:count_nerf]        #  0415 -2          

        if(1):#针对 maicity 数据集
            source_save=np.zeros((points_nerf.shape[0],3),dtype=float)
            for i in range(0, points_nerf.shape[0]):     #判断是否是有效点云     
                source_save[i][0]=points_nerf[i][0]
                source_save[i][1]=points_nerf[i][1]         
                source_save[i][2]=points_nerf[i][2]         
            if(j==data_start):
                source_save_all=source_save
            else:                   
                source_save_all   = np.concatenate([source_save_all, source_save],axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(source_save_all)                
            # file_path = os.path.join(save_path, 'source_select_tmp.pcd')  
            
            file_path = os.path.join(save_path, 'source_'+str(select_id)+'.pcd')                          
            # file_path = os.path.join(save_path, 'source_753.pcd')              
            o3d.io.write_point_cloud(file_path, pcd)

        if(1):# 保存各帧的 pose
            pose_save=np.ones((4,1),dtype=float)
            pose_save[0][0]=poses[j+1][0,-1]
            pose_save[1][0]=poses[j+1][1,-1]       
            pose_save[2][0]=poses[j+1][2,-1]           # 修改 0617                     
            pose_save=T_start_inv @ pose_save
            pose_save = pose_save.T[:,:3]                     
            
            pose_save_all   = np.concatenate([pose_save_all, pose_save],axis=0)
            pcd_pose = o3d.geometry.PointCloud()
            pcd_pose.points = o3d.utility.Vector3dVector(pose_save_all)                
            # file_path = os.path.join(save_path, 'pose_select_tmp.pcd')  
            file_path = os.path.join(save_path, 'pose_'+str(select_id)+'.pcd')              
            o3d.io.write_point_cloud(file_path, pcd_pose)

    print("训练集大小:",count_train)
    ### 神经辐射场空间范围
    source_pointcloud_path =  os.path.join(save_path, 'source_'+str(select_id)+'.pcd')
    pcd = o3d.io.read_point_cloud(source_pointcloud_path)
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    print("神经辐射场宽度范围: [",min_bound[1],", ",max_bound[1],"]")         
    print("神经辐射场长度范围: [",min_bound[0],", ",max_bound[0],"]")                 
    print("神经辐射场高度范围: [",min_bound[2],", ",max_bound[2],"]")           


def distance_to_ray(ray_origin, ray_dir, points):
    """ 计算射线起点到点集中每个点的距离 """
    v = points - ray_origin
    dist = np.sqrt(np.sum(v ** 2, axis=1))
    v_normalized = v / dist[:, np.newaxis]
    cos_angle = np.sum(v_normalized * ray_dir, axis=1)# 这两个向量都已经归一化了
    sin_angle = np.sqrt(1 - cos_angle ** 2)
    dist_to_ray = dist * sin_angle
    dist_to_ray = np.where(cos_angle > 0, dist_to_ray, np.inf) # 射线夹角小于等于0的情况，基本上不会与射线相交
    return dist_to_ray

def voxel_ray_casting_inference(voxel_size=0.5, select_id=0, save_path=None):
    # 1441~1445.pcd, 去除 1443.pcd, 用于构建点云地图
    map_pcd = o3d.io.read_point_cloud(save_path+"source.pcd")    
    map_pcd_np = np.asarray(map_pcd.points)
    # print("点云编号:",select_id,"体素大小:",voxel_size)
    # print("地图点云数目:",map_pcd_np.shape[0])

    # 对点云地图进行 # 体素化
    # 将点云划分到一个个体素内部，并计算每个体素内部点云坐标均值作为新的点云
    # voxel_size = 0.5
    downsampled_point_cloud = map_pcd.voxel_down_sample(voxel_size)
    voxel_centers = np.asarray(downsampled_point_cloud.points)
    # print("体素化地图点云数目:",voxel_centers.shape[0])

    voxel_centers_pcd = o3d.geometry.PointCloud()
    voxel_centers_pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    
    if voxel_size==0.05:
        o3d.io.write_point_cloud(save_path+"reconstruction_raycast/source_voxel005"+".pcd", voxel_centers_pcd)                   
    if voxel_size==0.1:
        o3d.io.write_point_cloud(save_path+"reconstruction_raycast/source_voxel01"+".pcd", voxel_centers_pcd)                           
    if voxel_size==0.25:
        o3d.io.write_point_cloud(save_path+"reconstruction_raycast/source_voxel025"+".pcd", voxel_centers_pcd)                      
    if voxel_size==0.5:
        o3d.io.write_point_cloud(save_path+"reconstruction_raycast/source_voxel05"+".pcd", voxel_centers_pcd)                      
    if voxel_size==0.75:
        o3d.io.write_point_cloud(save_path+"reconstruction_raycast/source_voxel075"+".pcd", voxel_centers_pcd)                      

    # 1443.pcd, 用于测试 
    test_pcd = o3d.io.read_point_cloud(save_path+"source_"+str(select_id)+".pcd")        
    test_pcd_np = np.asarray(test_pcd.points)
    # print("测试点云数目:",test_pcd_np.shape[0])
    if 0:
        mask = (test_pcd_np[:, 0] >= 4) #  会出现同时打在行人和车辆上的情况    
        test_pcd_np = test_pcd_np[mask]
    # print("选定测试点云数目:",test_pcd_np.shape[0])    
    test_pcd = o3d.geometry.PointCloud()
    test_pcd.points = o3d.utility.Vector3dVector(test_pcd_np)

    ###2. 获取被测试点云点云的位姿
    pose_select_pcd = o3d.io.read_point_cloud(save_path+"pose_"+str(select_id)+".pcd")        
    positions_test = (np.asarray(pose_select_pcd.points)).squeeze()

    num = test_pcd_np.shape[0]
    origin = np.tile(positions_test, (num, 1))
    direction = test_pcd_np - origin
    direction_length = np.linalg.norm(direction, axis=1)
    direction_normalized = direction / direction_length[:, np.newaxis]
    threshold_distance = voxel_size # 体素化

    inferred_pcd_np = test_pcd_np.copy()
    for i in tqdm(range(0, num)):          
        dist_to_ray = distance_to_ray(origin[i], direction_normalized[i], voxel_centers) #   体素化    
        near_point = voxel_centers[dist_to_ray <= threshold_distance, :]  #   体素化          
        threshold_distance_tmp = threshold_distance
        while near_point.shape[0]==0:
            threshold_distance_tmp = threshold_distance_tmp +voxel_size  # 体素化            
            near_point = voxel_centers[dist_to_ray <= threshold_distance_tmp, :]  # 体素化               
        inferred_pcd_np[i] = np.mean(near_point, axis=0)
        if(abs(inferred_pcd_np[i][0]-origin[i][0])<1 and abs(inferred_pcd_np[i][1]-origin[i][1])<1):
            print("i:",i)

    inferred_pcd = o3d.geometry.PointCloud()
    inferred_pcd.points = o3d.utility.Vector3dVector(inferred_pcd_np)
    if voxel_size==0.05:
        o3d.io.write_point_cloud(save_path+"reconstruction_raycast/voxel_ray_casting_inference/voxel005_"+str(select_id)+".pcd", inferred_pcd)                        
    if voxel_size==0.1:
        o3d.io.write_point_cloud(save_path+"reconstruction_raycast/voxel_ray_casting_inference/voxel01_"+str(select_id)+".pcd", inferred_pcd)                            
    if voxel_size==0.25:
        o3d.io.write_point_cloud(save_path+"reconstruction_raycast/voxel_ray_casting_inference/voxel025_"+str(select_id)+".pcd", inferred_pcd)                            
    if voxel_size==0.5:
        o3d.io.write_point_cloud(save_path+"reconstruction_raycast/voxel_ray_casting_inference/voxel05_"+str(select_id)+".pcd", inferred_pcd)                            
    if voxel_size==0.75:
        o3d.io.write_point_cloud(save_path+"reconstruction_raycast/voxel_ray_casting_inference/voxel075_"+str(select_id)+".pcd", inferred_pcd)                            


def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1
    Args:
        nx3 np.array's
    Returns:
        ([indices], [distances])
    """
    # print("verts1.shape: ",verts1.shape)
    # print("verts2.shape: ",verts2.shape)    
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        # Search_knn_vector_3d，返回查询点的K个最近邻索引列表，并将这些相邻的点存储在一个数组中。 https://blog.csdn.net/weixin_48138515/article/details/121643761
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)#参考https://zhuanlan.zhihu.com/p/462008591
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances

# def eval_pts(pts1, pts2, threshold=0.5):
# 实验配置 42 
def eval_pts(pts1, pts2, threshold=0.2):
    _, dist1 = nn_correspondance(pts1, pts2)
    _, dist2 = nn_correspondance(pts2, pts1)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist1<threshold).astype('float'))
    recall = np.mean((dist2<threshold).astype('float'))
    fscore = 2 * precision * recall / (precision + recall)
    cd = np.mean(dist1) + np.mean(dist2)

    return cd, fscore

def abs_error(pred, gt):
    value = np.abs(pred - gt)
    return np.mean(value)#求所有元素的平均值

def acc_thres(pred, gt,threshold=0.2):
    error = np.abs(pred - gt)
    # acc_thres = error < 0.2    
    acc_thres = error < threshold          
    acc_thres = np.sum(acc_thres) / acc_thres.shape[0] * 100
    return acc_thres

def error_metrics(voxel_size=0.5, select_id=0,threshold=0, save_path=None):
    gt_pcd_path = save_path+"source_"+str(select_id)+".pcd"
    gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)            
    gt_pcd_np = np.asarray(gt_pcd.points)
    # 加载激光雷达坐标原点 =========================================
    pose_select_pcd = o3d.io.read_point_cloud(save_path+"pose_"+str(select_id)+".pcd")    
    positions_test = (np.asarray(pose_select_pcd.points)).squeeze()
    num = gt_pcd_np.shape[0]
    origin = np.tile(positions_test, (num, 1))
    gt_vec = gt_pcd_np - origin # 计算每一行到点的向量
    gt_dist_vec = np.linalg.norm(gt_vec, axis=1) # 计算每一行到点的距离值    

    if voxel_size==0.75:
        pcd_path = save_path+"reconstruction_raycast/voxel_ray_casting_inference/voxel075_"+str(select_id)+".pcd"        
        pred_pcd = o3d.io.read_point_cloud(pcd_path)        
    if voxel_size==0.5:
        pcd_path = save_path+"reconstruction_raycast/voxel_ray_casting_inference/voxel05_"+str(select_id)+".pcd"        
        pred_pcd = o3d.io.read_point_cloud(pcd_path)                
    if voxel_size==0.25:
        pcd_path = save_path+"reconstruction_raycast/voxel_ray_casting_inference/voxel025_"+str(select_id)+".pcd"        
        pred_pcd = o3d.io.read_point_cloud(pcd_path)                
    if voxel_size==0.1:
        pcd_path = save_path+"reconstruction_raycast/voxel_ray_casting_inference/voxel01_"+str(select_id)+".pcd"        
        pred_pcd = o3d.io.read_point_cloud(pcd_path)                
    if voxel_size==0.05:
        pcd_path = save_path+"reconstruction_raycast/voxel_ray_casting_inference/voxel005_"+str(select_id)+".pcd"        
        pred_pcd = o3d.io.read_point_cloud(pcd_path)                

    pred_pcd_np = np.asarray(pred_pcd.points)    
    cd, fscore = eval_pts(pred_pcd_np, gt_pcd_np, threshold=threshold)
    # print("cd:",cd,"fscore:",fscore)       
    pred_vec = pred_pcd_np - origin # 计算每一行到点的向量
    pred_dist_vec = np.linalg.norm(pred_vec, axis=1) # 计算每一行到点的距离值
    abs_error_ = abs_error(pred_dist_vec, gt_dist_vec)
    acc_thres_ = acc_thres(pred_dist_vec, gt_dist_vec,threshold=threshold)
    # print("点云编号:",select_id)
    # print("cd:",cd,"fscore:",fscore, "abs_error_:",abs_error_,"acc_thres_:",acc_thres_)               
    # print("cd:","fscore:", "abs_error_:","acc_thres_:")             
    # print(cd,"     ",fscore, "      ",abs_error_,"      ",acc_thres_)
    # print(("\t{:>8}" * 4).format("Avg. Error", "Acc", "CD", "F"))#                     
    print(("\t{: 8.6f}" * 4).format(abs_error_, acc_thres_, cd, fscore))    

    return cd, fscore, abs_error_, acc_thres_

def multi_frame_pointcloud_fusion_maicity(root_dir, data_start=1,data_end=2,
                            range_delete_x=2, range_delete_y=1, range_delete_z=0.5,
                            nerf_length_min=-4.5, nerf_length_max=25.5, nerf_width_min=-12, nerf_width_max=12,nerf_height_min=-2, nerf_height_max=0.5,
                            pose_path="/home/meng/subject/maicity_dataset/01/poses.txt",
                            save_path="/home/meng/subject/maicity数据预处理/maicity数据配置0511/",select_id=0):
    # ### 1.神经辐射场空间范围初始化
    # print("神经辐射场宽度范围: [",nerf_width_min,", ",nerf_width_max,"]")         
    # print("神经辐射场长度范围: [",nerf_length_min,", ",nerf_length_max,"]")                 
    # print("神经辐射场高度范围: [",nerf_height_min,", ",nerf_height_max,"]")                  

    ###2. 获取每帧点云的位姿和位置信息
    poses=[]
    file = open(pose_path, "r", encoding="utf-8")
    rows = file.readlines()# 读取文件中的所有行数据
    new_list = [row.strip() for row in rows]# 读取文件中的所有行数据
    for row in new_list:# 遍历文件中的所有行数据
        P_tmp=np.array([[0,0,0,1]])
        P_array = np.append(np.array([float(i) for i in row.strip('\n').split(' ')]).reshape(3,4), P_tmp, axis=0)
        poses.append(P_array)    
    file.close()
    poses=np.array(poses)
    positions=poses[: ,:3,-1]
    positions=positions
    poses = torch.Tensor(poses)    

    # 3. 神经辐射场空间感兴趣区域内有效数据采样
    source_save_all=np.ones((0,3))
    pose_save_all =np.ones((0,3))    
       
    # for j in range(data_start, data_end):    # 配置文件中 j 从0开始
    count_train = 0
    # for j in tqdm(range(data_start, data_end)):    # 配置文件中 j 从0开始        
    for j in range(data_start, data_end):    # 配置文件中 j 从0开始                
        # if((j+1-3)%5!=0): # 测试集每5帧选取一帧，相应训练集每5帧选取4帧，第一帧从3.pcd开始
        if(j+1==select_id):
            count_train = count_train + 1
        else:
            continue             
        file_path = os.path.join(root_dir, '{}.pcd'.format(j+1))                     
        # print("点云 file_path: ",file_path)           
        # print("j+1: ",j+1)               
        points_source = pcl.PointCloud()    
        points_source = pcl.load(file_path)#这里的激光雷达点云数据应该是以激光雷达为坐标系的
        
        if 1: 
            #  (1) 判断是否是有效点云
            points_source_numpy = points_source.to_array()
            mask1 = np.logical_or.reduce((np.abs(points_source_numpy[:, 0]) >= range_delete_x, 
                                        np.abs(points_source_numpy[:, 1]) >= range_delete_y,
                                        np.abs(points_source_numpy[:, 2]) >= range_delete_z))
            points_effective2 = points_source_numpy[mask1]      # 去除车体内部的点云
            dist = np.linalg.norm(points_effective2, axis=1) 
            mask3 = dist < 120 
            points_effective3 = points_effective2[mask3]  # 去除距离原点过远的点
            ones_arr = np.ones((1, points_effective3.shape[0]))
            points_effective = np.vstack((points_effective3.T, ones_arr))        # 将 n*3 形状的数组转换成 4*n 形状
            # print("有效点云数量: ",points_effective.shape)                       
            # (2) 将点云转换到全局坐标系中
            # print("poses[", j ,"]:\n",poses[j])           
            points_effective = poses[j] @ points_effective  # 结果为 tensor
            points_effective = points_effective.T[:,:3]              
            
            # (3) 逐个判断当前点云在哪个子场的包围框内, 并计算边界
            mask4 = (points_effective[:, 0] >= nerf_length_min) & (points_effective[:, 1] >= nerf_width_min) & (points_effective[:, 2] >= nerf_height_min) & \
                (points_effective[:, 0] <= nerf_length_max) & (points_effective[:, 1] <= nerf_width_max) & (points_effective[:, 2] <= nerf_height_max)
            points_nerf = points_effective[mask4]  # 去除不属于神经辐射场母场的点云
            # print("母场点云数量: ",points_nerf.shape)        
                    
        if(1):#针对 maicity 数据集
            source_save=np.zeros((points_nerf.shape[0],3),dtype=float)
            for i in range(0, points_nerf.shape[0]):     #判断是否是有效点云     
                source_save[i][0]=points_nerf[i][0]
                source_save[i][1]=points_nerf[i][1]         
                source_save[i][2]=points_nerf[i][2]             
            source_save_all   = np.concatenate([source_save_all, source_save],axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(source_save_all)          
            file_path = os.path.join(save_path, 'source_'+str(select_id)+'.pcd')                                        
            # file_path = os.path.join(save_path, 'source_tmp.pcd')  
            o3d.io.write_point_cloud(file_path, pcd)

        if(1):# 保存各帧的 pose
            pose_save=np.zeros((1,3),dtype=float)
            pose_save[0][0]=poses[j][0,-1]
            pose_save[0][1]=poses[j][1,-1]       
            pose_save[0][2]=poses[j][2,-1]           
            pose_save_all   = np.concatenate([pose_save_all, pose_save],axis=0)
            pcd_pose = o3d.geometry.PointCloud()
            pcd_pose.points = o3d.utility.Vector3dVector(pose_save_all)                
            file_path = os.path.join(save_path, 'pose_'+str(select_id)+'.pcd')                    
            # file_path = os.path.join(save_path, 'pose_tmp.pcd')  
            o3d.io.write_point_cloud(file_path, pcd_pose)
    
    # print("训练集大小:",count_train)


def multi_frame_pointcloud_fusion_kitti(root_dir, data_start=1,data_end=2,
                            range_delete_x=2, range_delete_y=1, range_delete_z=0.5,
                            interest_x=12, interest_y=6, # 针对单帧激光雷达设置感兴趣区域范围                             
                            pose_path="/home/meng/subject/maicity_dataset/01/poses.txt",
                            save_path="/home/meng/subject/maicity数据预处理/maicity数据配置0511/",
                            over_height=0.168,over_low=-2,select_id=0):

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
    positions=poses[: ,:3,-1]
    positions=positions

    # 求 data_start 时刻矩阵的逆矩阵
    # T_start=poses[data_start]
    T_start=poses[data_start+1]     # 修改 0617  
    T_start_inv = np.linalg.inv(T_start)
    T_start_inv = torch.from_numpy(T_start_inv).float()        
    poses = torch.Tensor(poses)    
    poses_start= T_start_inv @ poses # 将全局坐标系下的 pose 转换到 起始帧坐标系下

    # 3. 神经辐射场空间感兴趣区域内有效数据采样
    source_save_all=np.ones((0,3))
    pose_save_all =np.ones((0,3))    
    count_train = 0
    for j in range(data_start, data_end):    # 配置文件中 j 从0开始
        # if((j+1-3)%5!=0): # 测试集每5帧选取一帧，相应训练集每5帧选取4帧，第一帧从3.pcd开始
        if((j+1)==select_id):             
            count_train = count_train + 1
        else:
            continue                  

        # if((j+1)==1178): # 测试集每5帧选取一帧，相应训练集每5帧选取4帧，第一帧从3.pcd开始
        #     count_train = count_train + 1
        # else:
        #     continue              
        
        file_path = os.path.join(root_dir, '{}.pcd'.format(j+1))                     
        print("点云 file_path: ",file_path)           
        points_source = pcl.PointCloud()    
        points_source = pcl.load(file_path)#这里的激光雷达点云数据应该是以激光雷达为坐标系的
        print("原始点云数量: ",points_source.size)
        
        # 判断是否是有效点云
        count_effective=0       
        points_effective_tmp=np.zeros((4, points_source.size),dtype=float)                             
        for i in range(0, points_source.size):     
            # 在激光雷达坐标系中进行判断，去除车体内部的点云    
            if(abs(points_source[i][0])<range_delete_x and abs(points_source[i][1])<range_delete_y and abs(points_source[i][2])<range_delete_z):                                     
                continue   
            # 去除过高的点云
            if(points_source[i][2]>over_height) or (points_source[i][2]<over_low):
                continue
            # 去除过远的点云
            range_tmp=np.sqrt(np.square(points_source[i][0])+np.square(points_source[i][1])+np.square(points_source[i][2]))
            if(range_tmp>120):
                continue        
              
            points_effective_tmp[0][count_effective]=points_source[i][0]                      
            points_effective_tmp[1][count_effective]=points_source[i][1]         
            points_effective_tmp[2][count_effective]=points_source[i][2]          
            points_effective_tmp[3][count_effective]=1#补全       
            count_effective=count_effective+1
        print("有效点云数量: ",count_effective)                
        points_effective=points_effective_tmp[:, :count_effective]              
        
        # 将点云转换到全局坐标系中
        points_effective = poses_start[j+1] @ points_effective    # 修改 0617                        
        points_effective = points_effective.T[:,:3]              
        
        count_nerf=0 # 属于神经辐射场的点云计数
        points_nerf_tmp=np.zeros((points_effective.shape[0], 3),dtype=float)                      
        for i in range(0, points_effective.shape[0]):        
            # 选取在车辆移动路径附近的点云，过远的点云舍弃
            near_all_pose=False # 初始化不靠近车辆行驶路径
            for k in range(data_start, data_end):
                # if(abs(points_effective[i][0]-poses_start[k][0,-1])>interest_x or abs(points_effective[i][1]-poses_start[k][1,-1])>interest_y ):               
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
        print("在神经辐射场内的点云数量: ",count_nerf)                
        # points_nerf=points_nerf_tmp[:, :count_nerf]        #  0415 -2                        
        points_nerf=points_nerf_tmp[:count_nerf]        #  0415 -2          

        if(1):#针对 maicity 数据集
            source_save=np.zeros((points_nerf.shape[0],3),dtype=float)
            for i in range(0, points_nerf.shape[0]):     #判断是否是有效点云     
                source_save[i][0]=points_nerf[i][0]
                source_save[i][1]=points_nerf[i][1]         
                source_save[i][2]=points_nerf[i][2]         
            if(j==data_start):
                source_save_all=source_save
            else:                   
                source_save_all   = np.concatenate([source_save_all, source_save],axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(source_save_all)                
            # file_path = os.path.join(save_path, 'source_select_tmp.pcd')  
            # file_path = os.path.join(save_path, 'source_tmp.pcd')              
            file_path = os.path.join(save_path, "source_"+str(select_id)+"_.pcd")                          
            o3d.io.write_point_cloud(file_path, pcd)

        if(1):# 保存各帧的 pose
            pose_save=np.ones((4,1),dtype=float)
            pose_save[0][0]=poses[j+1][0,-1]
            pose_save[1][0]=poses[j+1][1,-1]       
            pose_save[2][0]=poses[j+1][2,-1]           # 修改 0617                     
            pose_save=T_start_inv @ pose_save
            pose_save = pose_save.T[:,:3]                     
            
            pose_save_all   = np.concatenate([pose_save_all, pose_save],axis=0)
            pcd_pose = o3d.geometry.PointCloud()
            pcd_pose.points = o3d.utility.Vector3dVector(pose_save_all)                
            # file_path = os.path.join(save_path, 'pose_select_tmp.pcd')  
            # file_path = os.path.join(save_path, 'pose_tmp.pcd')              
            file_path = os.path.join(save_path, "pose_"+str(select_id)+"_.pcd")                      
            o3d.io.write_point_cloud(file_path, pcd_pose)

    print("训练集大小:",count_train)
    ### 神经辐射场空间范围
    source_pointcloud_path =  os.path.join(save_path, "source_"+str(select_id)+"_.pcd")
    pcd = o3d.io.read_point_cloud(source_pointcloud_path)
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    print("神经辐射场宽度范围: [",min_bound[1],", ",max_bound[1],"]")         
    print("神经辐射场长度范围: [",min_bound[0],", ",max_bound[0],"]")                 
    print("神经辐射场高度范围: [",min_bound[2],", ",max_bound[2],"]")     


if __name__ == '__main__':
    if 0: 
        Lidar_height=0.168 #激光雷达高度, 地面不起伏的话, 只需要取该值的一般半, 这里略微进行放大
        data_start=1150       
        data_end=1200    # 这个不错        
        root_dir="/home/biter/paper2/kitti/pcd_remove_dynamic1150_1200"
        save_path = "/home/biter/paper2/sequence00_new0627/1151_1200_view/"
        kwargs = {'root_dir': root_dir, 'data_start':data_start, 'data_end':data_end, 
                            'range_delete_x':3, 'range_delete_y':2, 'range_delete_z':1.25,
                            'interest_x':20,'interest_y':20, # 针对单帧激光雷达设置感兴趣区域范围 
                            'pose_path':"/home/biter/paper2/kitti/poses.txt",
                            'save_path':save_path,
                            'over_height':Lidar_height, 'over_low':-2   }   
    if 0:
        Lidar_height=0.168 #激光雷达高度, 地面不起伏的话, 只需要取该值的一般半, 这里略微进行放大
        data_start=1000       
        data_end=1050    # 这个不错        
        root_dir="/home/biter/paper2/kitti/01/pcd_remove_dynamic"
        save_path = "/home/biter/paper2/sequence01_big25/1001_1050/"
        kwargs = {'root_dir': root_dir, 'data_start':data_start, 'data_end':data_end, 
                            'range_delete_x':3, 'range_delete_y':2, 'range_delete_z':1.25,
                            'interest_x':20,'interest_y':20, # 针对单帧激光雷达设置感兴趣区域范围 
                            'pose_path':"/home/biter/paper2/kitti/01/poses.txt",
                            'save_path':save_path,
                            'over_height':Lidar_height, 'over_low':-2
                            }   
    if 0:
        Lidar_height=0.168 #激光雷达高度, 地面不起伏的话, 只需要取该值的一般半, 这里略微进行放大
        data_start=175       
        data_end=225    # 这个不错        
        root_dir="/home/biter/paper2/kitti/02/pcd_remove_dynamic"
        save_path = "/home/biter/paper2/kitti_data/sequence02/176_225/"
        kwargs = {'root_dir': root_dir, 'data_start':data_start, 'data_end':data_end, 
                            'range_delete_x':3, 'range_delete_y':2, 'range_delete_z':1.25,
                            'interest_x':20,'interest_y':20, # 针对单帧激光雷达设置感兴趣区域范围 
                            'pose_path':"/home/biter/paper2/kitti/02/poses.txt",
                            'save_path':save_path,
                            'over_height':Lidar_height, 'over_low':-2
                            }           
    if 0:
        Lidar_height=0.168 #激光雷达高度, 地面不起伏的话, 只需要取该值的一般半, 这里略微进行放大
        data_start=750       
        data_end=800    # 这个不错        
        root_dir="/home/biter/paper2/kitti/03/pcd_remove_dynamic"
        save_path = "/home/biter/paper2/sequence03_big/751_800/"
        kwargs = {'root_dir': root_dir, 'data_start':data_start, 'data_end':data_end, 
                            'range_delete_x':3, 'range_delete_y':2, 'range_delete_z':1.25,
                            'interest_x':20,'interest_y':20, # 针对单帧激光雷达设置感兴趣区域范围 
                            'pose_path':"/home/biter/paper2/kitti/03/poses.txt",
                            'save_path':save_path,
                            'over_height':Lidar_height, 'over_low':-2
                            }   
    if 0:
        Lidar_height=0.168 #激光雷达高度, 地面不起伏的话, 只需要取该值的一般半, 这里略微进行放大
        data_start=50       
        data_end=100    # 这个不错        
        root_dir="/home/biter/paper2/kitti/04/pcd_remove_dynamic"
        save_path = "/home/biter/paper2/kitti_data/sequence04/51_100/"
        kwargs = {'root_dir': root_dir, 'data_start':data_start, 'data_end':data_end, 
                            'range_delete_x':3, 'range_delete_y':2, 'range_delete_z':1.25,
                            'interest_x':20,'interest_y':20, # 针对单帧激光雷达设置感兴趣区域范围 
                            'pose_path':"/home/biter/paper2/kitti/04/poses.txt",
                            'save_path':save_path,
                            'over_height':Lidar_height, 'over_low':-2
                            }           
    if 0:
        Lidar_height=0.168 #激光雷达高度, 地面不起伏的话, 只需要取该值的一般半, 这里略微进行放大
        data_start=2125       
        data_end=2175    # 这个不错        
        root_dir="/home/biter/paper2/kitti/05/pcd_remove_dynamic"
        save_path = "/home/biter/paper2/kitti_data/sequence05/2126_2175/"
        kwargs = {'root_dir': root_dir, 'data_start':data_start, 'data_end':data_end, 
                            'range_delete_x':3, 'range_delete_y':2, 'range_delete_z':1.25,
                            'interest_x':20,'interest_y':20, # 针对单帧激光雷达设置感兴趣区域范围 
                            'pose_path':"/home/biter/paper2/kitti/05/poses.txt",
                            'save_path':save_path,
                            'over_height':Lidar_height, 'over_low':-2
                            }           
    if 0:
        Lidar_height=0.168 #激光雷达高度, 地面不起伏的话, 只需要取该值的一般半, 这里略微进行放大
        data_start=435       
        data_end=485    # 这个不错        
        root_dir="/home/biter/paper2/kitti/06/pcd_remove_dynamic"
        save_path = "/home/biter/paper2/kitti_data/sequence06/436_485/"        
        kwargs = {'root_dir': root_dir, 'data_start':data_start, 'data_end':data_end, 
                            'range_delete_x':3, 'range_delete_y':2, 'range_delete_z':1.25,
                            'interest_x':20,'interest_y':20, # 针对单帧激光雷达设置感兴趣区域范围 
                            'pose_path':"/home/biter/paper2/kitti/06/poses.txt",
                            'save_path':save_path,
                            'over_height':Lidar_height, 'over_low':-2
                            }   
    if 0:
        Lidar_height=0.168 #激光雷达高度, 地面不起伏的话, 只需要取该值的一般半, 这里略微进行放大
        data_start=350       
        data_end=400    # 这个不错        
        root_dir="/home/biter/paper2/kitti/07/pcd_remove_dynamic"
        save_path = "/home/biter/paper2/kitti_data/sequence07/351_400/"
        kwargs = {'root_dir': root_dir, 'data_start':data_start, 'data_end':data_end, 
                            'range_delete_x':3, 'range_delete_y':2, 'range_delete_z':1.25,
                            'interest_x':20,'interest_y':20, # 针对单帧激光雷达设置感兴趣区域范围 
                            'pose_path':"/home/biter/paper2/kitti/07/poses.txt",
                            'save_path':save_path,
                            'over_height':Lidar_height, 'over_low':-2
                            }   
    if 0:
        Lidar_height=0.168 #激光雷达高度, 地面不起伏的话, 只需要取该值的一般半, 这里略微进行放大
        data_start=1025       
        data_end=1075    # 这个不错        
        root_dir="/home/biter/paper2/kitti/08/pcd_remove_dynamic"
        save_path = "/home/biter/paper2/kitti_data/sequence08/1026_1075/"
        kwargs = {'root_dir': root_dir, 'data_start':data_start, 'data_end':data_end, 
                            'range_delete_x':3, 'range_delete_y':2, 'range_delete_z':1.25,
                            'interest_x':20,'interest_y':20, # 针对单帧激光雷达设置感兴趣区域范围 
                            'pose_path':"/home/biter/paper2/kitti/08/poses.txt",
                            'save_path':save_path,
                            'over_height':Lidar_height, 'over_low':-2
                            }   
    if 0:
        Lidar_height=0.168 #激光雷达高度, 地面不起伏的话, 只需要取该值的一般半, 这里略微进行放大
        data_start=125       
        data_end=175    # 这个不错        
        root_dir="/home/biter/paper2/kitti/09/pcd_remove_dynamic"
        save_path = "/home/biter/paper2/kitti_data/sequence09/126_175/"
        kwargs = {'root_dir': root_dir, 'data_start':data_start, 'data_end':data_end, 
                            'range_delete_x':3, 'range_delete_y':2, 'range_delete_z':1.25,
                            'interest_x':20,'interest_y':20, # 针对单帧激光雷达设置感兴趣区域范围 
                            'pose_path':"/home/biter/paper2/kitti/09/poses.txt",
                            'save_path': save_path,
                            'over_height':Lidar_height, 'over_low':-2
                            }   
    if 1:
        Lidar_height=0.168 #激光雷达高度, 地面不起伏的话, 只需要取该值的一般半, 这里略微进行放大
        data_start=375       
        data_end=425    # 这个不错        
        root_dir="/home/biter/paper2/kitti/10/pcd_remove_dynamic"
        save_path = "/home/biter/paper2/kitti_data/sequence10/376_425/"
        kwargs = {'root_dir': root_dir, 'data_start':data_start, 'data_end':data_end, 
                            'range_delete_x':3, 'range_delete_y':2, 'range_delete_z':1.25,
                            'interest_x':20,'interest_y':20, # 针对单帧激光雷达设置感兴趣区域范围 
                            'pose_path':"/home/biter/paper2/kitti/10/poses.txt",
                            'save_path':save_path,
                            'over_height':Lidar_height, 'over_low':-2
                            }   


    print("save_path:",save_path,"=========================================================================")
    print("测试点云:")                
    for j in range(data_start, data_end):    # 配置文件中 j 从0开始        
        if((j+1-3)%5==0): # 测试集每5帧选取一帧，相应训练集每5帧选取4帧，第一帧从3.pcd开始
            print(j+1, end="    ")                         
    print("")
    if 0:      
        count_train = 0
        for j in range(data_start, data_end):    # 配置文件中 j 从0开始        
            if((j+1-3)%5==0): # 测试集每5帧选取一帧，相应训练集每5帧选取4帧，第一帧从3.pcd开始
                count_train = count_train + 1
                select_id_1=j+1
                print("select_id_1:",select_id_1)
            else:
                continue             
            multi_frame_pointcloud_fusion_kitti(**kwargs, select_id=select_id_1)

    voxel_size = 0.05
    if 0:
        if 0:
            start_time = time.time()  # 记录程序开始执行的时间        
            print("生成推理点云, 体素大小为:", voxel_size)
            for j in range(data_start, data_end):    # 配置文件中 j 从0开始        
                if((j+1-3)%5==0): # 测试集每5帧选取一帧，相应训练集每5帧选取4帧，第一帧从3.pcd开始
                    select_id_1=j+1
                else:
                    continue      
                print("当前测试点云:", select_id_1)                             
                # 被测点云与体素化的融合点云相交并得出推理点云==============================================
                voxel_ray_casting_inference(voxel_size=voxel_size, select_id=select_id_1, save_path = save_path)
            end_time = time.time()  # 记录程序执行结束的时间        
            execution_time = end_time - start_time  # 计算程序执行时间
            print(f"程序执行时间为 {execution_time} 秒")            
            print(f"每个 scan 渲染时间为 {execution_time/10} 秒")                         

        if 1:
            threshold = 0.2
            print("进行评估, 体素大小为:", voxel_size,"阈值为:",threshold)            
            metrics_np = np.zeros((110,4))    
            count_train = 0
            cd_sum =0
            fscore_sum =0
            abs_error__sum =0
            acc_thres__sum = 0    
            print(("\t{:>8}" * 4).format("Avg. Error", "Acc", "CD", "F"))#                         
            for j in range(data_start, data_end):    # 配置文件中 j 从0开始        
                if((j+1-3)%5==0): # 测试集每5帧选取一帧，相应训练集每5帧选取4帧，第一帧从 3.pcd 开始
                    select_id_1=j+1
                    count_train = count_train + 1            
                else:
                    continue           
                cd, fscore, abs_error_, acc_thres_ = error_metrics(voxel_size=voxel_size, select_id=select_id_1,threshold=threshold, save_path = save_path)
                cd_sum = cd_sum + cd 
                fscore_sum = fscore_sum + fscore       
                abs_error__sum = abs_error__sum + abs_error_ 
                acc_thres__sum = acc_thres__sum + acc_thres_                          
            print("计算均值: ====================================================================================")
            print(("\t{: 8.6f}" * 4).format(abs_error__sum/count_train, acc_thres__sum/count_train, cd_sum/count_train, fscore_sum/count_train))    
            # np.save(save_path+"reconstruction_raycast/"+"voxel0.5_metric_tmp",metrics_np)     

            threshold = 1
            print("进行评估, 体素大小为:", voxel_size,"阈值为:",threshold)            
            metrics_np = np.zeros((110,4))    
            count_train = 0
            cd_sum =0
            fscore_sum =0
            abs_error__sum =0
            acc_thres__sum = 0    
            print(("\t{:>8}" * 4).format("Avg. Error", "Acc", "CD", "F"))#                         
            for j in range(data_start, data_end):    # 配置文件中 j 从0开始        
                if((j+1-3)%5==0): # 测试集每5帧选取一帧，相应训练集每5帧选取4帧，第一帧从3.pcd开始
                    select_id_1=j+1
                    count_train = count_train + 1            
                else:
                    continue           
                cd, fscore, abs_error_, acc_thres_ = error_metrics(voxel_size=voxel_size, select_id=select_id_1,threshold=threshold, save_path = save_path)
                cd_sum = cd_sum + cd 
                fscore_sum = fscore_sum + fscore       
                abs_error__sum = abs_error__sum + abs_error_ 
                acc_thres__sum = acc_thres__sum + acc_thres_                          
            print("计算均值: ====================================================================================")
            print(("\t{: 8.6f}" * 4).format(abs_error__sum/count_train, acc_thres__sum/count_train, cd_sum/count_train, fscore_sum/count_train))    
            # np.save(save_path+"reconstruction_raycast/"+"voxel0.5_metric_tmp",metrics_np)     

