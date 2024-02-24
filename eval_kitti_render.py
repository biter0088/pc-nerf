import argparse
from torch.utils.data import DataLoader
import numpy as np
import struct
import sys
import open3d as o3d
import os
import math
import pcl # 

from tqdm import tqdm
import time

from nof.networks import Embedding, NOF,NOF_fine,NOF_coarse
from nof.render import render_rays_view_0525_2_2
from nof.nof_utils import load_ckpt, decode_batch,decode_batch2
from nof.criteria.metrics import *

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_path', type=str, default=None,
                        help='')       
    parser.add_argument('--test_data_create', type=int, default=0,
                        help='test_data_create = 0: use the test data in the dir;  test_data_create = 1: recreate the test data in the dir;')      
    parser.add_argument('--depth_inference_method', type=int, default=2,
                        help='depth_inference_method: one-step or two-step')      
    parser.add_argument('--dataset', type=str, default='maicity',
                        help='data set')       
    # data
    parser.add_argument('--root_dir', type=str,  default='~/ir-mcl/data/ipblab',   help='root directory of dataset')
    parser.add_argument('--subnerf_path', type=str, default=None,  help='path for saving point cloud from child nerf aabb box')          
    parser.add_argument('--parentnerf_path', type=str, default=None,   help='path for saving point cloud from parent nerf aabb box')     
    parser.add_argument('--over_height', type=float, default=0.168,
                        help='')    
    parser.add_argument('--over_low', type=float, default=0.168,
                        help='')        
    parser.add_argument('--interest_x', type=float, default=12,   help='around vehicle in the x direction')    
    parser.add_argument('--interest_y', type=float, default=10,   help='around vehicle in the y direction')        
    parser.add_argument('--view_pcd_number', type=int, default=1178,
                        help='')        
    parser.add_argument('--sub_nerf_test_num', type=int, default=3,
                        help='the number of child nerf aabb box')    
    parser.add_argument('--nerf_length_min', type=float, default=-4.5,
                        help='')        
    parser.add_argument('--nerf_length_max', type=float, default= 25.5,
                        help=')')            
    parser.add_argument('--nerf_width_min', type=float, default=-4.5,
                        help=')')        
    parser.add_argument('--nerf_width_max', type=float, default= 25.5,
                        help='')           

    parser.add_argument('--width_min', type=float, default=1.6,
                        help='')    
    parser.add_argument('--width_max', type=float, default=8,
                        help='')    
    parser.add_argument('--height_min', type=float, default=2.0,
                        help='')    
    parser.add_argument('--height_max', type=float, default=10,
                        help='')         
    parser.add_argument('--length_max', type=float, default=12,
                        help='')    
    parser.add_argument('--sub_nerf_length', type=float, default=3,
                        help='')        
    parser.add_argument('--length_delta', type=int, default=4,
                        help='')    
    parser.add_argument('--nerf_height_min', type=float, default=-2.0,
                        help='')    
    parser.add_argument('--nerf_height_max', type=float, default=0.5,
                        help='')
    parser.add_argument('--resolution_up', type=float, default=0.5,
                        help='')    
    parser.add_argument('--resolution_front', type=float, default=0.5,
                        help='')                

    parser.add_argument('--range_delete_x', type=float, default=2,
                        help='')
    parser.add_argument('--range_delete_y', type=float, default=1,
                        help='')    
    parser.add_argument('--range_delete_z', type=float, default=0.5,
                        help='')                              
    parser.add_argument('--range_delta', type=float, default=0.2,
                        help='')          

    parser.add_argument('--data_start', type=int, default=1,
                        help='')
    parser.add_argument('--data_end', type=int, default=2,
                        help='')    
    parser.add_argument('--truncationCompensation', type=int, default=0,
                        help='')            
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--pcd_path', type=str, default=None,
                        help='pretrained child nerf checkpoint path to load')            
    parser.add_argument('--metrics_path', type=str, default=None,
                        help='pretrained child nerf checkpoint path to load')            
    parser.add_argument('--child_ckpt_path', type=str, default=None,
                        help='pretrained child nerf checkpoint path to load')    
    parser.add_argument('--chunk', type=int, default=32 * 1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of fine samples')                        
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=0.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='std dev of noise added to regularize sigma')
    parser.add_argument('--L_pos', type=int, default=10,
                        help='the frequency of the positional encoding.')

    parser.add_argument('--max_range', type=float, default=6,
                        help='')
    parser.add_argument('--max_range_effective', type=float, default=40,
                        help='')     
    parser.add_argument('--max_range_train', type=float, default=8,
                        help='')
    parser.add_argument('--max_range_val', type=float, default=6,
                        help='')                                                                   
    parser.add_argument('--min_range', type=float, default=1.5,
                        help='')
    parser.add_argument('--cloud_size_val', type=int, default=128,
                        help='')
    parser.add_argument('--pose_path', type=str, default='/media/bit/T7/dataset/kitti/dataset/sequences/00/poses.txt',
                        help='')               
    parser.add_argument('--feature_size', type=int, default=256,
                        help='the dimension of the feature maps.')
    parser.add_argument('--use_skip', default=False, action="store_true",
                        help='use skip architecture')

    return parser.parse_args()


def error_metrics(pred, gt, rays, valid_mask_gt):
    # ranges to pointclouds
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]
    pred_pts = rays_o + rays_d * pred.unsqueeze(-1)
    gt_pts = rays_o + rays_d * gt.unsqueeze(-1)   

    abs_error_ = abs_error(pred, gt, valid_mask_gt)
    acc_thres_ = acc_thres(pred, gt, valid_mask_gt)
    cd, fscore = eval_points(pred_pts, gt_pts, valid_mask_gt)

    return abs_error_, acc_thres_, cd, fscore

def summary_errors(errors,metrics_path):
    print("\nError Evaluation")
    errors = torch.Tensor(errors)

    # Mean Errors
    mean_errors = errors.mean(0)
    np.save(metrics_path,mean_errors) #
    
    print(("\t{:>8}" * 4).format("Avg. Error", "Acc", "CD", "F"))
    print(("\t{: 8.2f}" * 4).format(*mean_errors.tolist()))

    print("\n-> Done!")

def vector_angle(x, y):
	Lx = np.sqrt(x.dot(x))
	Ly = (np.sum(y ** 2, axis=1)) ** (0.5)
	cos_angle = np.sum(x * y, axis=1) / (Lx * Ly)
	angle = np.arccos(cos_angle)
	angle2 = angle * 360 / 2 / np.pi
	return angle2


def compute_far_bound0429(p, d, p_min, p_max):
    distance_effective=[]# 
    for i in range(3):

        if(d[i]*(p_min[i]-p[i])>0):#
            distance=(p_min[i]-p[i])/d[i]
            p_end=p+distance*d#
            count=0
            for k in range(3):#
                if k==i:
                    continue
                if(p_end[k]>=p_min[k] and p_end[k]<=p_max[k]):
                    count=count+1
            if(count>=2):#
                distance_effective.append(distance)
                
        if(d[i]*(p_max[i]-p[i])>0):#
            distance=(p_max[i]-p[i])/d[i]
            p_end=p+distance*d#
            count=0
            for k in range(3):#
                if k==i:
                    continue
                if(p_end[k]>=p_min[k] and p_end[k]<=p_max[k]):
                    count=count+1
            if(count>=2):#
                distance_effective.append(distance)
    
    if(len(distance_effective)<2 or len(distance_effective)>2):
        intersectFlag=False
        distance_near=0
        distance_far=0
        return intersectFlag, distance_near, distance_far

    if(len(distance_effective)==2):
        intersectFlag=True       
             
        distance_near= distance_effective[0]                                                            
        distance_far= distance_effective[1]    
        if(distance_near>distance_far):
            distance_near, distance_far = distance_far, distance_near                                      
        return intersectFlag, distance_near, distance_far

def ray_aabb_distances(ray_origin, ray_dirs, aabb_min, aabb_max):

    t1 = (aabb_min[0] - ray_origin[0]) / ray_dirs[:, 0]
    t2 = (aabb_max[0] - ray_origin[0]) / ray_dirs[:, 0]
    tmin_x = np.minimum(t1, t2)
    tmax_x = np.maximum(t1, t2)
    print("here")

    t3 = (aabb_min[1] - ray_origin[1]) / ray_dirs[:, 1]
    t4 = (aabb_max[1] - ray_origin[1]) / ray_dirs[:, 1]
    tmin_y = np.minimum(t3, t4)
    tmax_y = np.maximum(t3, t4)

    t5 = (aabb_min[2] - ray_origin[2]) / ray_dirs[:, 2]
    t6 = (aabb_max[2] - ray_origin[2]) / ray_dirs[:, 2]
    tmin_z = np.minimum(t5, t6)
    tmax_z = np.maximum(t5, t6)

    tmin = np.max(np.vstack((tmin_x, tmin_y, tmin_z)), axis=0)
    tmax = np.min(np.vstack((tmax_x, tmax_y, tmax_z)), axis=0)

    dists = np.where(tmax >= tmin, tmax, np.inf)  ####   
    return dists

def distance_to_ray(ray_origin, ray_dir, points):
    # v = points - ray_origin
    v = points - ray_origin.numpy()    
    dist = np.sqrt(np.sum(v ** 2, axis=1))
    cos_angle = np.sum(v * ray_dir, axis=1) / dist
    sin_angle = np.sqrt(1 - cos_angle ** 2)
    dist_to_ray = dist * sin_angle
    return dist_to_ray

def multi_frame_maicity(root_dir, split='test', data_start=1,data_end=2,
                            range_delete_x=2, range_delete_y=1, range_delete_z=0.5,
                            sub_nerf_test_num=4,
                            nerf_length_min=-4.5, nerf_length_max=25.5, nerf_width_min=-12, nerf_width_max=12,nerf_height_min=-2, nerf_height_max=0.5,
                            pose_path="/home/meng/subject/maicity_dataset/01/poses.txt",
                            subnerf_path=None,
                            view_pcd_number=0, result_path = None,depth_inference_method=2):
    print("depth_inference_method:",depth_inference_method)    
    print("[",nerf_length_min,", ",nerf_length_max,"]")                  
    print("[",nerf_width_min,", ",nerf_width_max,"]")         
    print("[",nerf_height_min,", ",nerf_height_max,"]")                  

    poses=[]
    file = open(pose_path, "r", encoding="utf-8")
    rows = file.readlines()# 
    new_list = [row.strip() for row in rows]# 
    for row in new_list:# 
        P_tmp=np.array([[0,0,0,1]])
        P_array = np.append(np.array([float(i) for i in row.strip('\n').split(' ')]).reshape(3,4), P_tmp, axis=0)
        poses.append(P_array)    
    file.close()
    poses=np.array(poses)
    positions=poses[: ,:3,-1]
    # positions=positions
    poses = torch.Tensor(poses)    
    positions = torch.Tensor(positions)        
    # print("poses:",poses)        

    sub_nerf_bound=np.zeros((sub_nerf_test_num,6)) # 
    sub_nerf_bound_larger=np.zeros((sub_nerf_test_num,6)) 
    sub_nerf_center_point=np.zeros((sub_nerf_test_num,3))                
    for i in range(sub_nerf_test_num):    
        file_name = f"{i+1}.pcd"
        file_path = os.path.join(subnerf_path, file_name)
        pcd = o3d.io.read_point_cloud(file_path)
        bbox = pcd.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        extend_tmp=0.025      
        sub_nerf_bound[i][0]=min_bound[0]-extend_tmp
        sub_nerf_bound[i][1]=min_bound[1]-extend_tmp            
        sub_nerf_bound[i][2]=min_bound[2]-extend_tmp
        sub_nerf_bound[i][3]=max_bound[0]+extend_tmp           
        sub_nerf_bound[i][4]=max_bound[1]+extend_tmp
        sub_nerf_bound[i][5]=max_bound[2]+extend_tmp      

        extend_tmp2 = 0.025    
        sub_nerf_bound_larger[i][0]=min_bound[0]-extend_tmp2
        sub_nerf_bound_larger[i][1]=min_bound[1]-extend_tmp2            
        sub_nerf_bound_larger[i][2]=min_bound[2]-extend_tmp2
        sub_nerf_bound_larger[i][3]=max_bound[0]+extend_tmp2           
        sub_nerf_bound_larger[i][4]=max_bound[1]+extend_tmp2
        sub_nerf_bound_larger[i][5]=max_bound[2]+extend_tmp2   
          
        sub_nerf_center_point[i][0]=(min_bound[0]+max_bound[0])/2.0
        sub_nerf_center_point[i][1]=(min_bound[1]+max_bound[1])/2.0            
        sub_nerf_center_point[i][2]=(min_bound[2]+max_bound[2])/2.0               
    np.set_printoptions(linewidth=np.inf)                
    print("sub_nerf_bound:",sub_nerf_bound)            

    source_save_all=np.ones((0,3))
    pose_save_all =np.ones((0,3))    
    
    rays_intersect_all=torch.zeros((0,14),dtype=float)              
    other_interest_sub_nerf_number_all=torch.zeros((0,1),dtype=int)                  

    for j in range(data_start, data_end):    # 
        file_path = os.path.join(root_dir, '{}.pcd'.format(j+1))                     
        if ((j+1==view_pcd_number)):   #               
            print("LiDAR point cloud frame file_path: ",file_path)     
        else:
            continue        
         
        points_source = pcl.PointCloud()    
        points_source = pcl.load(file_path)#
        print("source point cloud number: ",points_source.size)

        points_source_numpy = points_source.to_array()
        # remove point cloud in the range of vehicle
        mask1 = np.logical_or.reduce((np.abs(points_source_numpy[:, 0]) >= range_delete_x, # 
                                    np.abs(points_source_numpy[:, 1]) >= range_delete_y,
                                    np.abs(points_source_numpy[:, 2]) >= range_delete_z))
        points_effective2 = points_source_numpy[mask1]      #   

        dist = np.linalg.norm(points_effective2, axis=1) # 
        mask3 = dist < 120 # 

        points_effective3 = points_effective2[mask3] 
        ones_arr = np.ones((1, points_effective3.shape[0]))
        points_effective = np.vstack((points_effective3.T, ones_arr))        # 
        print("effective point cloud number: ",points_effective.shape)              
        points_effective = poses[j] @ points_effective  # 
        points_effective = points_effective.T[:,:3]              
        
        # Determine which child nerf aabb box the current point cloud is within, and calculate the boundaries
        mask4 = (points_effective[:, 0] >= nerf_length_min) & (points_effective[:, 1] >= nerf_width_min) & (points_effective[:, 2] >= nerf_height_min) & \
            (points_effective[:, 0] <= nerf_length_max) & (points_effective[:, 1] <= nerf_width_max) & (points_effective[:, 2] <= nerf_height_max)
        points_effective = points_effective[mask4]      
        print("number of effective point cloud in parent NeRF",points_effective.shape)      
        
        vec = points_effective - np.array([positions[j][0], positions[j][1], positions[j][2]]) # 
        dist_vec = np.linalg.norm(vec, axis=1) # 
        dir_vec = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, vec) #         
        rays_intersect=np.zeros((0,14),dtype=float)     
        other_interest_sub_nerf_number = np.zeros((0,1),dtype=int)          
        count_nerf=0 # 
        points_nerf_tmp=np.zeros((points_effective.shape[0], 3),dtype=float)       
        parent_far_bound_dist=ray_aabb_distances(positions[j], dir_vec, 
                                                    np.array([nerf_length_min, nerf_width_min, nerf_height_min]), 
                                                    np.array([nerf_length_max, nerf_width_max, nerf_height_max]))

        # depth_inference_method = 2 #  two-step depth inference
        # depth_inference_method = 1 # one-step depth inference                        
        for i in tqdm(range(0, points_effective.shape[0])):                  
            if(1):                                         
                intersect_sub_nerf_num=0 # Initialize without intersecting any child nerf aabb box 
                # parent_near_bound = 1.0
                parent_near_bound = 0.0                
                parent_far_bound = parent_far_bound_dist[i]     
                rays_intersect_sub_nerf=np.zeros((sub_nerf_test_num,12),dtype=float)    #     

                center = (sub_nerf_bound[:, :3] + sub_nerf_bound[:, 3:]) / 2
                dist_to_ray = distance_to_ray(positions[j], dir_vec[i], center)
                sub_nerf_bound_filter = sub_nerf_bound_larger[dist_to_ray <= 0.65]  # 110940 和  110940                                  

                interest_or_not = False                               
                for k in range(sub_nerf_bound_filter.shape[0]):   # 
                    intersectFlag_tmp,near_bound_tmp, far_bound_tmp= compute_far_bound0429(positions[j],dir_vec[i], sub_nerf_bound_filter[k][:3], sub_nerf_bound_filter[k][3:6])      
                    if(intersectFlag_tmp==True):#
                        interest_or_not = True
                        if(near_bound_tmp>far_bound_tmp):
                            print("Error")
                            print("near_bound_tmp:",near_bound_tmp,"far_bound_tmp:",far_bound_tmp)                            
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][0]=positions[j][0]         # column 1~3 : ray_o
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][1]=positions[j][1]
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][2]=positions[j][2]                                                
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][3]=dir_vec[i][0]   # column 4~6 : ray_d
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][4]=dir_vec[i][1] 
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][5]=dir_vec[i][2]                           
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][6]=near_bound_tmp  # column 7: near_bound
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][7]=far_bound_tmp      # column 8: far_bound  
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][8]=3                                    # column 9: ray_class
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][9]=dist_vec[i]                # column 10 : range_readings
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][10]=parent_near_bound        # column 11 : parent_near_bound
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][11]=parent_far_bound            # column 12: parent_far_bound                        

                        intersect_sub_nerf_num=intersect_sub_nerf_num+1
                        
                        if depth_inference_method==1: # one-step depth inference: no need to calculate the intersection of rays with child nerf 
                            rays_intersect_sub_nerf[intersect_sub_nerf_num-1][6]=parent_near_bound  #  column 7: near_bound
                            rays_intersect_sub_nerf[intersect_sub_nerf_num-1][7]=parent_far_bound      # column 8: far_bound                              
                            break

                extend_iter = 0
                expand_signal = 0 #            
                while interest_or_not == False:
                    if extend_iter> 0.5: # 
                        expand_signal = 1
                        break                    
                    extend_iter = extend_iter + 0.005
                    sub_nerf_bound_filter[:, :3] = sub_nerf_bound_filter[:, :3] - extend_iter
                    sub_nerf_bound_filter[:, 3:6] = sub_nerf_bound_filter[:, 3:6] + extend_iter

                    for k in range(sub_nerf_bound_filter.shape[0]):   #
                        intersectFlag_tmp,near_bound_tmp, far_bound_tmp= compute_far_bound0429(positions[j],dir_vec[i], sub_nerf_bound_filter[k][:3], sub_nerf_bound_filter[k][3:6])      
                        if(intersectFlag_tmp==True):#
                            interest_or_not = True
                            if(near_bound_tmp>far_bound_tmp):
                                print("Error")
                                print("near_bound_tmp:",near_bound_tmp,"far_bound_tmp:",far_bound_tmp)                            
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][0]=positions[j][0]         
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][1]=positions[j][1]
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][2]=positions[j][2]                                                
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][3]=dir_vec[i][0]   
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][4]=dir_vec[i][1] 
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][5]=dir_vec[i][2]                           
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][6]=near_bound_tmp  
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][7]=far_bound_tmp     
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][8]=3                                    
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][9]=dist_vec[i]              
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][10]=parent_near_bound       
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][11]=parent_far_bound                         

                            intersect_sub_nerf_num=intersect_sub_nerf_num+1
                            
                            if depth_inference_method==1: # one-step depth inference: no need to calculate the intersection of rays with child nerf 
                                rays_intersect_sub_nerf[intersect_sub_nerf_num-1][6]=parent_near_bound  
                                rays_intersect_sub_nerf[intersect_sub_nerf_num-1][7]=parent_far_bound                            
                                break

                if expand_signal==1: #
                    continue

                rays_intersect_sub_nerf =rays_intersect_sub_nerf[:intersect_sub_nerf_num]            
                sorted_indices = np.argsort(rays_intersect_sub_nerf[:, 6]) #         
                rays_intersect_sub_nerf_order = rays_intersect_sub_nerf[sorted_indices]        
                row_num = np.arange(rays_intersect_sub_nerf_order.shape[0]).reshape(-1, 1)+1 # 
                new_rays_intersect_sub_nerf = np.hstack((rays_intersect_sub_nerf_order, row_num)) #  column 13: id Number of possible intersecting child nerf aabb box
                zeros_arr = -1 * np.ones((new_rays_intersect_sub_nerf.shape[0], 1))          #        
                rays_intersect_sub_nerf_end = np.concatenate((new_rays_intersect_sub_nerf, zeros_arr), axis=1) # column 14: other_interest_sub_nerf_number
                if(intersect_sub_nerf_num>=1):      
                    rays_intersect_sub_nerf_end[0][-1]=intersect_sub_nerf_num-1

                other_interest_sub_nerf_number_tmp = np.zeros((rays_intersect_sub_nerf_end.shape[0],1),dtype=int)  
                other_interest_sub_nerf_number_tmp[0]=intersect_sub_nerf_num-1

            if(intersect_sub_nerf_num==0): # 
                continue 

            rays_intersect=np.concatenate((rays_intersect, rays_intersect_sub_nerf_end), axis=0)       
            other_interest_sub_nerf_number=np.concatenate((other_interest_sub_nerf_number, other_interest_sub_nerf_number_tmp), axis=0)       ##        

            points_nerf_tmp[count_nerf][0]=points_effective[i][0]                      
            points_nerf_tmp[count_nerf][1]=points_effective[i][1]         
            points_nerf_tmp[count_nerf][2]=points_effective[i][2]        
            count_nerf=count_nerf+1
         
        rays_intersect = torch.Tensor(rays_intersect)     
        print("effective ray number: ",rays_intersect.shape[0])                
        rays_intersect_all = torch.cat([rays_intersect_all, rays_intersect], 0)
        other_interest_sub_nerf_number = torch.as_tensor(other_interest_sub_nerf_number)
        other_interest_sub_nerf_number_all = torch.cat([other_interest_sub_nerf_number_all, other_interest_sub_nerf_number], 0)        ## 

        print("point cloud number in child nerf aabb box:",count_nerf)                
        points_nerf=points_nerf_tmp[:, :count_nerf]       

        if depth_inference_method==2:
            directory_name = result_path+"/two_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/"
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)

            source_save=np.zeros((points_nerf.shape[0],3),dtype=float)
            for i in range(0, points_nerf.shape[0]):     #     
                source_save[i][0]=points_nerf[i][0]
                source_save[i][1]=points_nerf[i][1]         
                source_save[i][2]=points_nerf[i][2]             
            source_save_all   = np.concatenate([source_save_all, source_save],axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(source_save_all)                
            o3d.io.write_point_cloud(result_path+"/two_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/"+str(view_pcd_number)+"_source.pcd", pcd)

            pose_save=np.zeros((1,3),dtype=float)
            pose_save[0][0]=poses[j][0,-1]
            pose_save[0][1]=poses[j][1,-1]       
            pose_save[0][2]=poses[j][2,-1]           
            pose_save_all   = np.concatenate([pose_save_all, pose_save],axis=0)
            pcd_pose = o3d.geometry.PointCloud()
            pcd_pose.points = o3d.utility.Vector3dVector(pose_save_all)                
            o3d.io.write_point_cloud(result_path+"/two_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/"+str(view_pcd_number)+"_pose.pcd", pcd_pose)

        if depth_inference_method==1:
            directory_name = result_path+"/one_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/"
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)

            source_save=np.zeros((points_nerf.shape[0],3),dtype=float)
            for i in range(0, points_nerf.shape[0]):     #     
                source_save[i][0]=points_nerf[i][0]
                source_save[i][1]=points_nerf[i][1]         
                source_save[i][2]=points_nerf[i][2]             
            source_save_all   = np.concatenate([source_save_all, source_save],axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(source_save_all)                
            o3d.io.write_point_cloud(result_path+"/one_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/"+str(view_pcd_number)+"_source.pcd", pcd)

            pose_save=np.zeros((1,3),dtype=float)
            pose_save[0][0]=poses[j][0,-1]
            pose_save[0][1]=poses[j][1,-1]       
            pose_save[0][2]=poses[j][2,-1]      #       
            pose_save_all   = np.concatenate([pose_save_all, pose_save],axis=0)
            pcd_pose = o3d.geometry.PointCloud()
            pcd_pose.points = o3d.utility.Vector3dVector(pose_save_all)                
            o3d.io.write_point_cloud(result_path+"/one_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/"+str(view_pcd_number)+"_pose.pcd", pcd_pose)

    rays_intersect_all=rays_intersect_all.float()
    all_ranges = rays_intersect_all[:, 9:10]
    all_rays = torch.cat([rays_intersect_all[:, :9], rays_intersect_all[:, 10:]], dim=1)

    if depth_inference_method==2:    
        np.save(result_path+"/two_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/all_ranges_child.npy",all_ranges.data.cpu().numpy()) # 
        np.save(result_path+"/two_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/all_rays_child.npy",all_rays.data.cpu().numpy()) # 
        np.save(result_path+"/two_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/other_interest_sub_nerf_number_child.npy",
                    other_interest_sub_nerf_number_all.data.cpu().numpy()) # 
    if depth_inference_method==1:    
        np.save(result_path+"/one_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/all_ranges_child.npy",all_ranges.data.cpu().numpy()) # 
        np.save(result_path+"/one_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/all_rays_child.npy",all_rays.data.cpu().numpy()) # 
        np.save(result_path+"/one_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/other_interest_sub_nerf_number_child.npy",
                    other_interest_sub_nerf_number_all.data.cpu().numpy()) # 

    return all_rays, all_ranges,other_interest_sub_nerf_number_all # 0627        


def multi_frame_kitti(root_dir, split='test', data_start=1439, data_end=1510,
                            range_delete_x=2, range_delete_y=1, range_delete_z=0.5,
                            sub_nerf_test_num=4,
                            over_height = 0.168, #                           
                            over_low = -2, #                                             
                            interest_x=12, #
                            interest_y=10, #                                
                            pose_path=None,
                            subnerf_path=None,
                            parentnerf_path=None,
                            view_pcd_number=0, result_path = None,depth_inference_method=2):   #  
    print("depth_inference_method:",depth_inference_method)
    parent_nerf_pcd = o3d.io.read_point_cloud(parentnerf_path)
    parent_nerf_bbox = parent_nerf_pcd.get_axis_aligned_bounding_box()
    parent_nerf_min_bound = parent_nerf_bbox.get_min_bound()
    parent_nerf_max_bound = parent_nerf_bbox.get_max_bound()        
    nerf_length_min= parent_nerf_min_bound[0]
    nerf_width_min = parent_nerf_min_bound[1]        
    nerf_height_min = parent_nerf_min_bound[2]        
    nerf_length_max= parent_nerf_max_bound[0]              
    nerf_width_max = parent_nerf_max_bound[1]        
    nerf_height_max = parent_nerf_max_bound[2]           
    print("[",nerf_length_min,", ",nerf_length_max,"]")         
    print("[",nerf_width_min,", ",nerf_width_max,"]")         
    print("[",nerf_height_min,", ",nerf_height_max,"]")                  

    T_velo2cam=np.array([[4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02],
                                    [-7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02],
                                    [9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01],
                                    [0,0,0,1]])#    
    poses=[]
    file = open(pose_path, "r", encoding="utf-8")
    rows = file.readlines()# 
    new_list = [row.strip() for row in rows]# 
    for row in new_list:# 
        P_tmp=np.array([[0,0,0,1]])
        P_array = np.append(np.array([float(i) for i in row.strip('\n').split(' ')]).reshape(3,4), P_tmp, axis=0)
        P_array= np.matmul(P_array, T_velo2cam)#                  
        poses.append(P_array)    
    file.close()
    poses=np.array(poses)

    T_start=poses[data_start+1] 
    T_start_inv = np.linalg.inv(T_start)
    T_start_inv = torch.from_numpy(T_start_inv).float()        
    poses = torch.Tensor(poses)    
    poses= T_start_inv @ poses 
    positions=poses[: ,:3,-1]

    sub_nerf_bound=np.zeros((sub_nerf_test_num,6)) 
    print("subnerf_path:",subnerf_path)     
    for i in range(sub_nerf_test_num):    
        file_name = f"{i+1}.pcd"
        file_path = os.path.join(subnerf_path, file_name)
        pcd = o3d.io.read_point_cloud(file_path)
        bbox = pcd.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        # extend_tmp=0.05
        extend_tmp=0    
        sub_nerf_bound[i][0]=min_bound[0]-extend_tmp
        sub_nerf_bound[i][1]=min_bound[1]-extend_tmp            
        sub_nerf_bound[i][2]=min_bound[2]-extend_tmp
        sub_nerf_bound[i][3]=max_bound[0]+extend_tmp           
        sub_nerf_bound[i][4]=max_bound[1]+extend_tmp
        sub_nerf_bound[i][5]=max_bound[2]+extend_tmp        
    print("sub_nerf_bound.shape:",sub_nerf_bound.shape)             

    source_save_all=np.ones((0,3))
    pose_save_all =np.ones((0,3))    
    
    rays_intersect_all=torch.zeros((0,14),dtype=float)              
    true_in_all=torch.zeros((0,1),dtype=bool)                  
    other_interest_sub_nerf_number_all=torch.zeros((0,1),dtype=int)                  
    for j in range(data_start, data_end):    
        file_path = os.path.join(root_dir, '{}.pcd'.format(j+1))                     

        if ((j+1==view_pcd_number)):   #               
            print("LiDAR point cloud frame file_path: ",file_path)     
        else:
            continue        

        points_source = pcl.PointCloud()    
        points_source = pcl.load(file_path)#
        print("source point cloud number: ",points_source.size)
        points_source_numpy = points_source.to_array()
        
        mask1 = np.logical_or.reduce((np.abs(points_source_numpy[:, 0]) >= range_delete_x, #  
                                    np.abs(points_source_numpy[:, 1]) >= range_delete_y,
                                    np.abs(points_source_numpy[:, 2]) >= range_delete_z))
        points_effective1 = points_source_numpy[mask1]      
        print("point cloud number after removing the point cloud in the vehicle range: ",points_effective1.shape[0])        

        mask2 = points_effective1[:, 2] <= over_height  #  
        points_effective2 = points_effective1[mask2]
        mask2 = points_effective2[:, 2] >= over_low  #  
        points_effective2 = points_effective2[mask2]

        dist = np.linalg.norm(points_effective2, axis=1) #  
        mask3 = dist < 120 
        # mask3 = dist < 25
        points_effective3 = points_effective2[mask3] 
        
        ones_arr = np.ones((1, points_effective3.shape[0])) #  
        points_effective = np.vstack((points_effective3.T, ones_arr))       
        points_effective = poses[j+1] @ points_effective        #        
        points_effective4 = points_effective.T[:,:3]              
        
        if 1:
            count_nerf=0 # 
            points_nerf_tmp=np.zeros((points_effective4.shape[0], 3),dtype=float)                      
            for i in range(0, points_effective4.shape[0]):        
                near_all_pose=False # 
                for k in range(data_start, data_end):
                    if(abs(points_effective4[i][0]-poses[k+1][0,-1])>interest_x or abs(points_effective4[i][1]-poses[k+1][1,-1])>interest_y ):                                                
                        continue
                    else:
                        near_all_pose=True
                        break
                if(near_all_pose==False):
                    continue
                points_nerf_tmp[count_nerf][0]=points_effective4[i][0]                     
                points_nerf_tmp[count_nerf][1]=points_effective4[i][1]         
                points_nerf_tmp[count_nerf][2]=points_effective4[i][2]        
                count_nerf=count_nerf+1
            points_effective=points_nerf_tmp[:count_nerf]               

        vec = points_effective - np.array([positions[j+1][0], positions[j+1][1], positions[j+1][2]]) #        
        dist_vec = np.linalg.norm(vec, axis=1) # 
        dir_vec = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, vec) #         
        rays_intersect=np.zeros((0,14),dtype=float)     
        true_in=np.zeros((0,1),dtype=bool)     

        other_interest_sub_nerf_number = np.zeros((0,1),dtype=int)  
        count_nerf=0 # 
        points_nerf_tmp=np.zeros((points_effective.shape[0], 3),dtype=float)       

        parent_far_bound_dist=ray_aabb_distances(positions[j+1], dir_vec, 
                                                    np.array([nerf_length_min, nerf_width_min, nerf_height_min]), 
                                                    np.array([nerf_length_max, nerf_width_max, nerf_height_max]))    #

        # depth_inference_method = 2 #  two-step depth inference
        # depth_inference_method = 1 # one-step depth inference    
        for i in tqdm(range(0, points_effective.shape[0])):                  
            if(1):                                         
                intersect_sub_nerf_num=0 #  
                # parent_near_bound = 1.0
                parent_near_bound = 0.0                
                parent_far_bound = parent_far_bound_dist[i]     
                rays_intersect_sub_nerf=np.zeros((sub_nerf_test_num,12),dtype=float)    #     

                true_in_sub_nerf=np.zeros((sub_nerf_test_num,1),dtype=bool)    
                                
                center = (sub_nerf_bound[:, :3] + sub_nerf_bound[:, 3:]) / 2
                dist_to_ray = distance_to_ray(positions[j+1], dir_vec[i], center)            
                sub_nerf_bound_filter = sub_nerf_bound[dist_to_ray <= 0.65]  

                interest_or_not = False                               
                for k in range(sub_nerf_bound_filter.shape[0]):   
                    intersectFlag_tmp,near_bound_tmp, far_bound_tmp= compute_far_bound0429(positions[j+1],dir_vec[i], sub_nerf_bound_filter[k][:3], sub_nerf_bound_filter[k][3:6])                          
                    if(intersectFlag_tmp==True):#
                        interest_or_not = True
                        if(near_bound_tmp>far_bound_tmp):
                            print("Error")
                            print("near_bound_tmp:",near_bound_tmp,"far_bound_tmp:",far_bound_tmp)                            
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][0]=positions[j+1][0]        
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][1]=positions[j+1][1]
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][2]=positions[j+1][2]                                                                            
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][3]=dir_vec[i][0]  
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][4]=dir_vec[i][1] 
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][5]=dir_vec[i][2]                           
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][6]=near_bound_tmp  
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][7]=far_bound_tmp      
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][8]=3                                    
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][9]=dist_vec[i]              
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][10]=parent_near_bound        
                        rays_intersect_sub_nerf[intersect_sub_nerf_num][11]=parent_far_bound                           

                        if points_effective[i][0]>=sub_nerf_bound_filter[k][0] and points_effective[i][0]<=sub_nerf_bound_filter[k][3] and \
                            points_effective[i][1]>=sub_nerf_bound_filter[k][1] and points_effective[i][1]<=sub_nerf_bound_filter[k][4] and \
                            points_effective[i][2]>=sub_nerf_bound_filter[k][2] and points_effective[i][2]<=sub_nerf_bound_filter[k][5]:
                            true_in_sub_nerf[intersect_sub_nerf_num] = True

                        if parent_far_bound<far_bound_tmp:
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][11]=far_bound_tmp                                      

                        intersect_sub_nerf_num=intersect_sub_nerf_num+1
                        
                        if depth_inference_method==1: # one-step depth inference: no need to calculate the intersection of rays with child nerf 
                            rays_intersect_sub_nerf[intersect_sub_nerf_num-1][6]=parent_near_bound 
                            rays_intersect_sub_nerf[intersect_sub_nerf_num-1][7]=parent_far_bound                                
                            break
                        
                extend_iter = 0
                expand_signal = 0 
                while interest_or_not == False:
                    if extend_iter> 0.5: 
                        expand_signal = 1
                        break
                    extend_iter = extend_iter + 0.05                   
                    sub_nerf_bound_filter[:, :3] = sub_nerf_bound_filter[:, :3] - extend_iter
                    sub_nerf_bound_filter[:, 3:6] = sub_nerf_bound_filter[:, 3:6] + extend_iter
                    for k in range(sub_nerf_bound_filter.shape[0]):  
                        intersectFlag_tmp,near_bound_tmp, far_bound_tmp= compute_far_bound0429(positions[j+1],dir_vec[i], sub_nerf_bound_filter[k][:3], sub_nerf_bound_filter[k][3:6])                
                        if(intersectFlag_tmp==True):
                            interest_or_not = True
                            if(near_bound_tmp>far_bound_tmp):
                                print("Error")
                                print("near_bound_tmp:",near_bound_tmp,"far_bound_tmp:",far_bound_tmp)                            
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][0]=positions[j+1][0]        
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][1]=positions[j+1][1]
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][2]=positions[j+1][2]                                                       
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][3]=dir_vec[i][0]   
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][4]=dir_vec[i][1] 
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][5]=dir_vec[i][2]                           
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][6]=near_bound_tmp  
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][7]=far_bound_tmp      
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][8]=3                                    
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][9]=dist_vec[i]               
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][10]=parent_near_bound        
                            rays_intersect_sub_nerf[intersect_sub_nerf_num][11]=parent_far_bound                    

                            if points_effective[i][0]>=sub_nerf_bound_filter[k][0] and points_effective[i][0]<=sub_nerf_bound_filter[k][3] and \
                                points_effective[i][1]>=sub_nerf_bound_filter[k][1] and points_effective[i][1]<=sub_nerf_bound_filter[k][4] and \
                                points_effective[i][2]>=sub_nerf_bound_filter[k][2] and points_effective[i][2]<=sub_nerf_bound_filter[k][5]:
                                true_in_sub_nerf[intersect_sub_nerf_num] = True

                            if parent_far_bound<far_bound_tmp:
                                rays_intersect_sub_nerf[intersect_sub_nerf_num][11]=far_bound_tmp                                

                            intersect_sub_nerf_num=intersect_sub_nerf_num+1
                            
                            if depth_inference_method==1: # one-step depth inference: no need to calculate the intersection of rays with child nerf 
                                rays_intersect_sub_nerf[intersect_sub_nerf_num-1][6]=parent_near_bound  
                                rays_intersect_sub_nerf[intersect_sub_nerf_num-1][7]=parent_far_bound                                
                                break
                if expand_signal==1: 
                    continue

                rays_intersect_sub_nerf =rays_intersect_sub_nerf[:intersect_sub_nerf_num]            
                true_in_sub_nerf =true_in_sub_nerf[:intersect_sub_nerf_num]                     
                sorted_indices = np.argsort(rays_intersect_sub_nerf[:, 6])       
                rays_intersect_sub_nerf_order = rays_intersect_sub_nerf[sorted_indices]        
                true_in_sub_nerf_order = true_in_sub_nerf[sorted_indices]        

                row_num = np.arange(rays_intersect_sub_nerf_order.shape[0]).reshape(-1, 1)+1 
                new_rays_intersect_sub_nerf = np.hstack((rays_intersect_sub_nerf_order, row_num)) 
                zeros_arr = -1 * np.ones((new_rays_intersect_sub_nerf.shape[0], 1))          
                rays_intersect_sub_nerf_end = np.concatenate((new_rays_intersect_sub_nerf, zeros_arr), axis=1) 
                if(intersect_sub_nerf_num>=1):      
                    rays_intersect_sub_nerf_end[0][-1]=intersect_sub_nerf_num-1
                
                other_interest_sub_nerf_number_tmp = np.zeros((rays_intersect_sub_nerf_end.shape[0],1),dtype=int)  
                other_interest_sub_nerf_number_tmp[0]=intersect_sub_nerf_num-1

            if(intersect_sub_nerf_num==0): 
                continue 
            rays_intersect=np.concatenate((rays_intersect, rays_intersect_sub_nerf_end), axis=0)      
            true_in=np.concatenate((true_in, true_in_sub_nerf_order), axis=0)      

            other_interest_sub_nerf_number=np.concatenate((other_interest_sub_nerf_number, other_interest_sub_nerf_number_tmp), axis=0)                   

            points_nerf_tmp[count_nerf][0]=points_effective[i][0]                      
            points_nerf_tmp[count_nerf][1]=points_effective[i][1]         
            points_nerf_tmp[count_nerf][2]=points_effective[i][2]        
            count_nerf = count_nerf+1
         
        rays_intersect = torch.Tensor(rays_intersect)     
        true_in = torch.Tensor(true_in)             
        print("effective ray number: ",rays_intersect.shape[0])                
        rays_intersect_all = torch.cat([rays_intersect_all, rays_intersect], 0)
        true_in_all = torch.cat([true_in_all, true_in], 0)        
        other_interest_sub_nerf_number = torch.as_tensor(other_interest_sub_nerf_number)
        other_interest_sub_nerf_number_all = torch.cat([other_interest_sub_nerf_number_all, other_interest_sub_nerf_number], 0)        

        print("point cloud number in child nerf aabb box: ",count_nerf)                
        points_nerf=points_nerf_tmp[:, :count_nerf]       

        if depth_inference_method==2:
            directory_name = result_path+"/two_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/"
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)

            if(1):
                source_save=np.zeros((points_nerf.shape[0],3),dtype=float)
                for i in range(0, points_nerf.shape[0]):     #     
                    source_save[i][0]=points_nerf[i][0]
                    source_save[i][1]=points_nerf[i][1]         
                    source_save[i][2]=points_nerf[i][2]             
                source_save_all   = np.concatenate([source_save_all, source_save],axis=0)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(source_save_all)                
                o3d.io.write_point_cloud(result_path+"/two_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/"+str(view_pcd_number)+"_source.pcd", pcd)
            if(1):#  pose
                pose_save=np.zeros((1,3),dtype=float)
                pose_save[0][0]=poses[j+1][0,-1]
                pose_save[0][1]=poses[j+1][1,-1]       
                pose_save[0][2]=poses[j+1][2,-1]      #       
                pose_save_all   = np.concatenate([pose_save_all, pose_save],axis=0)
                pcd_pose = o3d.geometry.PointCloud()
                pcd_pose.points = o3d.utility.Vector3dVector(pose_save_all)                
                o3d.io.write_point_cloud(result_path+"/two_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/"+str(view_pcd_number)+"_pose.pcd", pcd_pose)

        if depth_inference_method==1:
            directory_name = result_path+"/one_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/"
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)

            if(1):
                source_save=np.zeros((points_nerf.shape[0],3),dtype=float)
                for i in range(0, points_nerf.shape[0]):     #     
                    source_save[i][0]=points_nerf[i][0]
                    source_save[i][1]=points_nerf[i][1]         
                    source_save[i][2]=points_nerf[i][2]             
                source_save_all   = np.concatenate([source_save_all, source_save],axis=0)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(source_save_all)                
                o3d.io.write_point_cloud(result_path+"/one_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/"+str(view_pcd_number)+"_source.pcd", pcd)
            if(1):#  pose
                pose_save=np.zeros((1,3),dtype=float)
                pose_save[0][0]=poses[j+1][0,-1]
                pose_save[0][1]=poses[j+1][1,-1]       
                pose_save[0][2]=poses[j+1][2,-1]      #      
                pose_save_all   = np.concatenate([pose_save_all, pose_save],axis=0)
                pcd_pose = o3d.geometry.PointCloud()
                pcd_pose.points = o3d.utility.Vector3dVector(pose_save_all)                
                o3d.io.write_point_cloud(result_path+"/one_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/"+str(view_pcd_number)+"_pose.pcd", pcd_pose)

    rays_intersect_all=rays_intersect_all.float()
    all_ranges = rays_intersect_all[:, 9:10]
    all_rays = torch.cat([rays_intersect_all[:, :9], rays_intersect_all[:, 10:]], dim=1)

    if depth_inference_method==2:    
        np.save(result_path+"/two_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/all_ranges_child.npy",all_ranges.data.cpu().numpy()) # 
        np.save(result_path+"/two_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/all_rays_child.npy",all_rays.data.cpu().numpy()) # 
        np.save(result_path+"/two_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/other_interest_sub_nerf_number_child.npy",
                    other_interest_sub_nerf_number_all.data.cpu().numpy()) # 
    if depth_inference_method==1:    
        np.save(result_path+"/one_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/all_ranges_child.npy",all_ranges.data.cpu().numpy()) # 
        np.save(result_path+"/one_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/all_rays_child.npy",all_rays.data.cpu().numpy()) # 
        np.save(result_path+"/one_step/"+str(view_pcd_number)+"pcd/childnerf_ray_intersect/other_interest_sub_nerf_number_child.npy",
                    other_interest_sub_nerf_number_all.data.cpu().numpy()) # 

    return all_rays, all_ranges,other_interest_sub_nerf_number_all # 0627        


if __name__ == '__main__':
    hparams = get_opts()

    if 1: # 
        use_cuda: bool = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
    print('Using: ', device)

    print("\nLoading Model and Data")
    embedding_position = Embedding(in_channels=3, N_freq=hparams.L_pos)# 
    nof_coarse_model = NOF_coarse(feature_size=hparams.feature_size,
                    in_channels_xy=3 + 3 * hparams.L_pos * 2,
                    use_skip=hparams.use_skip)
    nof_fine_model = NOF_fine(feature_size=hparams.feature_size,
                    in_channels_xy=3 + 3 * hparams.L_pos * 2,
                    use_skip=hparams.use_skip)
    nof_ckpt = hparams.ckpt_path
    print("ckpt:",hparams.ckpt_path)
    load_ckpt(nof_coarse_model, nof_ckpt, model_name='nof_coarse')
    load_ckpt(nof_fine_model, nof_ckpt, model_name='nof_fine')    
    nof_coarse_model.to(device).eval()
    nof_fine_model.to(device).eval()    

    if (hparams.child_ckpt_path!=None):
        child_nof_coarse_model = NOF_coarse(feature_size=hparams.feature_size,
                        in_channels_xy=3 + 3 * hparams.L_pos * 2,
                        use_skip=hparams.use_skip)
        child_nof_fine_model = NOF_fine(feature_size=hparams.feature_size,
                        in_channels_xy=3 + 3 * hparams.L_pos * 2,
                        use_skip=hparams.use_skip)
        # loading child nerf pretrained weights
        child_nof_ckpt = hparams.child_ckpt_path
        print("ckpt:",hparams.child_ckpt_path)
        load_ckpt(child_nof_coarse_model, child_nof_ckpt, model_name='nof_coarse')
        load_ckpt(child_nof_fine_model, child_nof_ckpt, model_name='nof_fine')    
        child_nof_coarse_model.to(device).eval()
        child_nof_fine_model.to(device).eval()    

    if(hparams.dataset=="maicity"):
        for j in range(hparams.data_start, hparams.data_end):            
            if((j+1-3-hparams.data_start)%5==0):                # frame sparsity = 20%
            # if((j+1-hparams.data_start)%4==0):                 # frame sparsity = 25%         
            # if((j+1-hparams.data_start)%3==0):                 # frame sparsity = 33%                      
            # if((j+1-hparams.data_start)%2==0):                 # frame sparsity = 50%
            # if((j+1-1-hparams.data_start)%3!=0):              # frame sparsity = 67%
            # if((j+1-1-hparams.data_start)%4!=0):              # frame sparsity = 75%      
            # if((j+1-3-hparams.data_start)%5!=0):              # frame sparsity = 80%
            # if((j+1-5-hparams.data_start)%10!=0):            # frame sparsity = 90%

                start_time = time.time()  #                            
                hparams.view_pcd_number=j+1 
                print("view_pcd_number:",hparams.view_pcd_number,"=====================================")

                kwargs = {'root_dir': hparams.root_dir, 'data_start':hparams.data_start, 'data_end':hparams.data_end, 
                                    'range_delete_x':hparams.range_delete_x, 'range_delete_y':hparams.range_delete_y, 'range_delete_z':hparams.range_delete_z,
                                    'sub_nerf_test_num':hparams.sub_nerf_test_num,
                                    'nerf_length_min':hparams.nerf_length_min, 'nerf_length_max':hparams.nerf_length_max,
                                    'nerf_width_min':hparams.nerf_width_min, 'nerf_width_max':hparams.nerf_width_max,
                                    'nerf_height_min':hparams.nerf_height_min, 'nerf_height_max':hparams.nerf_height_max, 
                                    'pose_path':hparams.pose_path,
                                    'subnerf_path':hparams.subnerf_path,
                                    'view_pcd_number':hparams.view_pcd_number,
                                    'result_path':hparams.result_path,                                                                            
                                    }

                test_data_create = hparams.test_data_create #  test_data_create = 0: use the test data in the dir;  test_data_create = 1: recreate the test data in the dir;  
                depth_inference_method = hparams.depth_inference_method #  2 indicate two-step Depth Inference,  and 1 indicate one-step Depth Inference
                result_path = hparams.result_path
               
                if test_data_create == 0:                    
                    if depth_inference_method==2:
                        dataset_ranges=np.load(result_path+"/two_step/"+str(hparams.view_pcd_number)+"pcd/childnerf_ray_intersect/all_ranges_child.npy")  
                        print("load test data"+result_path+"/two_step/"+str(hparams.view_pcd_number)+"pcd/childnerf_ray_intersect/all_ranges_child.npy")
                        dataset_rays=np.load(result_path+"/two_step/"+str(hparams.view_pcd_number)+"pcd/childnerf_ray_intersect/all_rays_child.npy")  
                        dataset_other_interest_sub_nerf_number=np.load(result_path+"/two_step/"+str(hparams.view_pcd_number)+"pcd/childnerf_ray_intersect/other_interest_sub_nerf_number_child.npy")      
                    else: # depth_inference_method==1
                        dataset_ranges=np.load(result_path+"/one_step/"+str(hparams.view_pcd_number)+"pcd/childnerf_ray_intersect/all_ranges_child.npy")  
                        print("load test data"+result_path+"/one_step/"+str(hparams.view_pcd_number)+"pcd/childnerf_ray_intersect/all_ranges_child.npy")
                        dataset_rays=np.load(result_path+"/one_step/"+str(hparams.view_pcd_number)+"pcd/childnerf_ray_intersect/all_rays_child.npy")  
                        dataset_other_interest_sub_nerf_number=np.load(result_path+"/one_step/"+str(hparams.view_pcd_number)+"pcd/childnerf_ray_intersect/other_interest_sub_nerf_number_child.npy")                              
                else:
                    dataset_rays, dataset_ranges, dataset_other_interest_sub_nerf_number = multi_frame_maicity(split='test',  **kwargs,  depth_inference_method=depth_inference_method)          
                                        
                print("-> test data load done !")

                if 1:
                    dataset_ranges = torch.tensor(dataset_ranges)
                    dataset_rays = torch.tensor(dataset_rays)
                    dataset_other_interest_sub_nerf_number = torch.tensor(dataset_other_interest_sub_nerf_number)            
                    print("dataset_rays.shape: ",dataset_rays.shape) 

                    errors = []
                    view_pointcloud=np.empty(shape=(0, 3))
                    view_pointcloud=torch.tensor(view_pointcloud).to(device)
                
                    batch_size_set=18432                                                                        
                    current_index=0       
                    pbar = tqdm(total=(dataset_rays.shape[0]/batch_size_set+1))    
                    i=0
                    other_ray_number=0
                    while i<dataset_rays.shape[0]:
                        if(i==dataset_rays.shape[0]-1): 
                            print("end")                    
                            break     

                        if(i+batch_size_set<dataset_rays.shape[0]-0.5*batch_size_set):                                   
                            other_ray_number=0
                            while dataset_rays[i+batch_size_set+other_ray_number,-1]< -0.5 : 
                                other_ray_number=other_ray_number+1
                                if(i+batch_size_set+other_ray_number==dataset_rays.shape[0]): 
                                    print("enter here 1")                    
                                    break
                            rays = dataset_rays[i: i+batch_size_set+other_ray_number, :]
                            ranges = dataset_ranges[i: i+batch_size_set+other_ray_number]       
                            other_interest_sub_nerf_number = dataset_other_interest_sub_nerf_number[i: i+batch_size_set+other_ray_number]               
                            i = i+batch_size_set+other_ray_number                   
                        else:
                            print("enter here 2")
                            rays = dataset_rays[i: dataset_rays.shape[0], :]
                            ranges = dataset_ranges[i: dataset_rays.shape[0]]  
                            other_interest_sub_nerf_number = dataset_other_interest_sub_nerf_number[i: dataset_rays.shape[0]]                 
                            i =  dataset_rays.shape[0]              

                        pbar.update(1)
                        pbar.set_description("Processing %d" % i)        
                        rays = rays.squeeze()
                        rays = rays.to(device)
                        ranges = ranges.squeeze()  # shape: (N_beams,)
                        ranges = ranges.to(device)#
                        other_interest_sub_nerf_number = other_interest_sub_nerf_number.squeeze()   
                        other_interest_sub_nerf_number = other_interest_sub_nerf_number.to(device)  
                        
                        with torch.no_grad():
                            rendered_rays = render_rays_view_0525_2_2(
                                model=nof_coarse_model, model_fine=nof_fine_model, 
                                embedding_xy=embedding_position, rays=rays,other_interest_sub_nerf_number=other_interest_sub_nerf_number,  
                                N_samples=hparams.N_samples, N_importance=hparams.N_importance, use_disp=hparams.use_disp, perturb=hparams.perturb,
                                noise_std=hparams.noise_std, chunk=hparams.chunk, depth_inference_method=depth_inference_method
                            )       

                        gt = ranges
                        points_inference_fine = rendered_rays['points_inference_fine']
                        pred_fine = rendered_rays['depth_fine']       #         
                        rays_effective_flag_fine = rendered_rays['rays_effective_flag_fine']
                        valid_mask_gt = torch.squeeze(rays_effective_flag_fine).bool()
                        points_inference_fine = points_inference_fine[valid_mask_gt, :]  # 
                        view_pointcloud=torch.cat((view_pointcloud, points_inference_fine),dim=0)        

                    pbar.close()

                    np_pcd = torch.tensor([item.cpu().detach().numpy() for item in view_pointcloud]).cuda()
                    print("np_pcd.shape:",np_pcd.shape)
                    np_pcd = np_pcd.cpu().numpy()
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(np_pcd)
                    print("len(pcd.points):",len(pcd.points))
                    print("render pcd path:",hparams.pcd_path+str(hparams.view_pcd_number))
                    if depth_inference_method==2:                    
                        o3d.io.write_point_cloud(hparams.pcd_path+str(hparams.view_pcd_number)+"_two_step.pcd", pcd)                
                    else:
                        o3d.io.write_point_cloud(hparams.pcd_path+str(hparams.view_pcd_number)+"_one_step.pcd", pcd)                           
                    # print("-> Done!")

                end_time = time.time()  #         
                execution_time = end_time - start_time  # 
                print(f"execution_time: {execution_time} s")            
            else:
                continue          

    if(hparams.dataset=="kitti"):
        for j in range(hparams.data_start, hparams.data_end):    
            if((j+1-3-hparams.data_start)%5==0):                # frame sparsity = 20%
            # if((j+1-hparams.data_start)%4==0):                 # frame sparsity = 25%         
            # if((j+1-hparams.data_start)%3==0):                 # frame sparsity = 33%                      
            # if((j+1-hparams.data_start)%2==0):                 # frame sparsity = 50%
            # if((j+1-1-hparams.data_start)%3!=0):              # frame sparsity = 67%
            # if((j+1-1-hparams.data_start)%4!=0):              # frame sparsity = 75%      
            # if((j+1-3-hparams.data_start)%5!=0):              # frame sparsity = 80%
            # if((j+1-5-hparams.data_start)%10!=0):            # frame sparsity = 90%

                start_time = time.time()  #                                                
                hparams.view_pcd_number=j+1 
                print("view_pcd_number:",hparams.view_pcd_number,"=====================================")        

                kwargs = {'root_dir': hparams.root_dir, 'data_start':hparams.data_start, 'data_end':hparams.data_end, 
                                    'range_delete_x':hparams.range_delete_x, 'range_delete_y':hparams.range_delete_y, 'range_delete_z':hparams.range_delete_z,
                                    'sub_nerf_test_num':hparams.sub_nerf_test_num,
                                    'pose_path':hparams.pose_path,
                                    'subnerf_path':hparams.subnerf_path,
                                    'parentnerf_path':hparams.parentnerf_path,  # 
                                    'over_height':hparams.over_height, 'over_low':hparams.over_low, 
                                    'interest_x':hparams.interest_x, 'interest_y':hparams.interest_y,  #         
                                    'view_pcd_number':hparams.view_pcd_number,
                                    'result_path':hparams.result_path,                                    
                                    }

                test_data_create = hparams.test_data_create #  test_data_create = 0: use the test data in the dir;  test_data_create = 1: recreate the test data in the dir;  
                depth_inference_method = hparams.depth_inference_method #  2 indicate two-step Depth Inference,  and 1 indicate one-step Depth Inference
                result_path = hparams.result_path

                if test_data_create == 0:
                    if depth_inference_method==2:
                        dataset_ranges=np.load(result_path+"/two_step/"+str(hparams.view_pcd_number)+"pcd/childnerf_ray_intersect/all_ranges_child.npy")  
                        print("load test data"+result_path+"/two_step/"+str(hparams.view_pcd_number)+"pcd/childnerf_ray_intersect/all_ranges_child.npy")
                        dataset_rays=np.load(result_path+"/two_step/"+str(hparams.view_pcd_number)+"pcd/childnerf_ray_intersect/all_rays_child.npy")  
                        dataset_other_interest_sub_nerf_number=np.load(result_path+"/two_step/"+str(hparams.view_pcd_number)+"pcd/childnerf_ray_intersect/other_interest_sub_nerf_number_child.npy")      
                    else: # depth_inference_method = 1
                        dataset_ranges=np.load(result_path+"/one_step/"+str(hparams.view_pcd_number)+"pcd/childnerf_ray_intersect/all_ranges_child.npy")  
                        print("load test data"+result_path+"/one_step/"+str(hparams.view_pcd_number)+"pcd/childnerf_ray_intersect/all_ranges_child.npy")
                        dataset_rays=np.load(result_path+"/one_step/"+str(hparams.view_pcd_number)+"pcd/childnerf_ray_intersect/all_rays_child.npy")  
                        dataset_other_interest_sub_nerf_number=np.load(result_path+"/one_step/"+str(hparams.view_pcd_number)+"pcd/childnerf_ray_intersect/other_interest_sub_nerf_number_child.npy")                              
                else:
                    dataset_rays, dataset_ranges,dataset_other_interest_sub_nerf_number = multi_frame_kitti(split='test',  **kwargs, depth_inference_method=depth_inference_method)             #

                print("-> test data load done!")

                if 1:
                    dataset_ranges = torch.tensor(dataset_ranges)
                    dataset_rays = torch.tensor(dataset_rays)
                    dataset_other_interest_sub_nerf_number = torch.tensor(dataset_other_interest_sub_nerf_number)            
                    print("dataset_rays.shape: ",dataset_rays.shape) 

                    errors = []
                    view_pointcloud=np.empty(shape=(0, 3))
                    view_pointcloud=torch.tensor(view_pointcloud).to(device)
                
                    print("\nSynthesis scans with NOF_coarse and NOF_fine")
                    batch_size_set=4096         
                    current_index=0      
                    pbar = tqdm(total=(dataset_rays.shape[0]/batch_size_set+1))    
                    i=0
                    other_ray_number=0
                    while i<dataset_rays.shape[0]:
                        if(i==dataset_rays.shape[0]-1): 
                            print("end")                    
                            break     
                        if(i+batch_size_set<dataset_rays.shape[0]-0.5*batch_size_set):            
                            other_ray_number=0
                            while dataset_rays[i+batch_size_set+other_ray_number,-1]< -0.5 : 
                                other_ray_number=other_ray_number+1
                                if(i+batch_size_set+other_ray_number==dataset_rays.shape[0]): 
                                    print("enter here 1")                    
                                    break
                            rays = dataset_rays[i: i+batch_size_set+other_ray_number, :]
                            ranges = dataset_ranges[i: i+batch_size_set+other_ray_number]       
                            other_interest_sub_nerf_number = dataset_other_interest_sub_nerf_number[i: i+batch_size_set+other_ray_number]                 
                            i = i+batch_size_set+other_ray_number                   
                        else:
                            print("enter here 2")
                            rays = dataset_rays[i: dataset_rays.shape[0], :]
                            ranges = dataset_ranges[i: dataset_rays.shape[0]]  
                            other_interest_sub_nerf_number = dataset_other_interest_sub_nerf_number[i: dataset_rays.shape[0]]                      
                            i =  dataset_rays.shape[0]              

                        pbar.update(1)
                        pbar.set_description("Processing %d" % i)        
                        rays = rays.squeeze()
                        rays = rays.to(device)
                        ranges = ranges.squeeze()  # shape: (N_beams,)
                        ranges = ranges.to(device)#
                        other_interest_sub_nerf_number = other_interest_sub_nerf_number.squeeze()   
                        other_interest_sub_nerf_number = other_interest_sub_nerf_number.to(device)  
                        
                        with torch.no_grad():
                            rendered_rays = render_rays_view_0525_2_2(
                                model=nof_coarse_model, model_fine=nof_fine_model, 
                                embedding_xy=embedding_position, rays=rays,other_interest_sub_nerf_number=other_interest_sub_nerf_number,  
                                N_samples=hparams.N_samples, N_importance=hparams.N_importance, use_disp=hparams.use_disp, perturb=hparams.perturb,
                                noise_std=hparams.noise_std, chunk=hparams.chunk, depth_inference_method=depth_inference_method
                            )       

                        gt = ranges
                        points_inference_fine = rendered_rays['points_inference_fine']
                        pred_fine = rendered_rays['depth_fine']       #         
                        rays_effective_flag_fine = rendered_rays['rays_effective_flag_fine']
                        valid_mask_gt = torch.squeeze(rays_effective_flag_fine).bool()
                        points_inference_fine = points_inference_fine[valid_mask_gt, :] 
                        view_pointcloud=torch.cat((view_pointcloud, points_inference_fine),dim=0)        

                    pbar.close()

                    np_pcd = torch.tensor([item.cpu().detach().numpy() for item in view_pointcloud]).cuda()
                    print("np_pcd.shape:",np_pcd.shape)
                    np_pcd = np_pcd.cpu().numpy()
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(np_pcd)
                    print("len(pcd.points):",len(pcd.points))
                    print("render pcd path:",hparams.pcd_path+str(hparams.view_pcd_number))
                    if depth_inference_method==2:                    
                        o3d.io.write_point_cloud(hparams.pcd_path+str(hparams.view_pcd_number)+"_two_step.pcd", pcd)                
                    else:
                        o3d.io.write_point_cloud(hparams.pcd_path+str(hparams.view_pcd_number)+"_one_step.pcd", pcd)                                        

                end_time = time.time()  #         
                execution_time = end_time - start_time  # 
                print(f"execution_time: {execution_time} s")            

