import os
import json
import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree
import torch
import torch.utils.data as data

import pcl # 
import math
import time

def repeat_ornot(x, y, index ,x_array, y_array):
    threshold=2.5
    if(index < 25): #一开始的一些点
        for i in range(index+25,len(x_array)):        
            if(abs(x-x_array[i])<=threshold and abs(y-y_array[i])<=threshold):
                return True

    elif(index >= 25 and index <len(x_array)-25 ): #中间的一些点
        for i in range(0,index-25):        
            if(abs(x-x_array[i])<=threshold and abs(y-y_array[i])<=threshold):
                return True
        for i in range(index+25,len(x_array)):        
            if(abs(x-x_array[i])<=threshold and abs(y-y_array[i])<=threshold):
                return True        

    elif(index >= len(x_array)-25):#结束的一些点
        for i in range(0,index-25):        
            if(abs(x-x_array[i])<=threshold and abs(y-y_array[i])<=threshold):
                return True        

    return False        


def compute_far_bound(ray_o, ray_d, x_max, x_min, y_max, y_min, z_max, z_min):

    # Calculate the intersection distance between the rays emitted from the origin and the six surfaces of the rectangular prism
    # ray_o: Ray origin coordinates
    # ray_d: ray direction vector, in units of vector
    # x_max, x_min, y_max, y_min, z_max, z_min: The coordinates of the six surfaces of a rectangular prism on the coordinate axis

    if ray_d[0] != 0:
        t1 = (x_max - ray_o[0]) / ray_d[0]
        t2 = (x_min - ray_o[0]) / ray_d[0]
        if t1 < 0:
            t1 = np.inf
        if t2 < 0:
            t2 = np.inf
    else:
        t1, t2 = np.inf, np.inf

    if ray_d[1] != 0:
        t3 = (y_max - ray_o[1]) / ray_d[1]
        t4 = (y_min - ray_o[1]) / ray_d[1]
        if t3 < 0:
            t3 = np.inf
        if t4 < 0:
            t4 = np.inf
    else:
        t3, t4 = np.inf, np.inf

    if ray_d[2] != 0:
        t5 = (z_max - ray_o[2]) / ray_d[2]
        t6 = (z_min - ray_o[2]) / ray_d[2]
        if t5 < 0:
            t5 = np.inf
        if t6 < 0:
            t6 = np.inf
    else:
        t5, t6 = np.inf, np.inf

    t = np.min([t1, t2, t3, t4, t5, t6])
    if t == np.inf:
        return None
    else:
        return t

"""
    1. Assuming the origin is not within the box, calculate which surface of the box intersects with the specified direction of rays emitted from the origin, and output the minimum and maximum distances
"""
def compute_far_bound0406(p, d, p_min, p_max):
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
    
    distance_near= distance_effective[0]                                                            
    distance_far= distance_effective[1]    
    if(distance_near>distance_far):
        distance_near, distance_far = distance_far, distance_near                                      

    return distance_near, distance_far

"""
    1. Assuming the origin is not within the box, calculate which surface of the box intersects with the specified direction of rays emitted from the origin, and output the minimum and maximum distances
"""
def compute_far_bound0606(p, d, p_min, p_max):
    # d=d.numpy()
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
    
    intersect = True
    if len(distance_effective)>2:
        distance_near= distance_effective[0]                                                            
        distance_far= distance_effective[0]    
        for i in range(len(distance_effective)):
            if(distance_near>distance_effective[i]):
                distance_near= distance_effective[i] 
            if(distance_far<distance_effective[i]):
                distance_far= distance_effective[i]                 
    
    if len(distance_effective)==2:
        distance_near= distance_effective[0]                                                            
        distance_far= distance_effective[1]    
        if(distance_near>distance_far):
            distance_near, distance_far = distance_far, distance_near                 
    if len(distance_effective)==1:# 
        distance_near= distance_effective[0]                                                            
        distance_far= distance_effective[0]       
        if(distance_near>distance_far):
            distance_near, distance_far = distance_far, distance_near         
                             
    if len(distance_effective)==0:    
        intersect = False       
        distance_near= 0                                                            
        distance_far= 0    
    return intersect, distance_near, distance_far

def find_aabb_box(points, aabb_list, query_point):
    """
    Use kdtree and aabb box search algorithms to find a spatial point inside which aabb box
        : param points: an array of coordinates for points, with a shape of (n, 3)
        : param aabb_list: AABB box array, with a shape of (n, 6), each row represents x_min, y_min, z_min, x_max, y_max, z_max, respectively
        : param query_point: The coordinates of the query point, with a shape of (3,)
        : return: A tuple, where the first element is a bool value, indicating whether the query point is inside any aabb box, and the second element is an int value, indicating the index of the aabb box where the query point is located,
        If the query point is not inside any aabb box, then the second element is None
    """
    tree = KDTree(points)
    distances, indices = tree.query(query_point.reshape(1, -1), k=10) 
    indices=indices.squeeze()
    aabb_indices = indices.tolist()
    nearest_aabb_index = None
    for i in aabb_indices:
        aabb = aabb_list[i]
        if query_point[0] >= aabb[0] and query_point[1] >= aabb[1] and query_point[2] >= aabb[2] \
                and query_point[0] <= aabb[3] and query_point[1] <= aabb[4] and query_point[2] <= aabb[5]:
            nearest_aabb_index = i
            break
    if nearest_aabb_index is None:
        return False, None
    else:
        return True, nearest_aabb_index


class maicity_dataload(data.Dataset):
    def __init__(self, root_dir, split='train', data_start=0,data_end=36,cloud_size_val=2048,
                            range_delete_x=2, range_delete_y=1, range_delete_z=0.5,
                            sub_nerf_test_num=3,
                            surface_expand=0.1,                            
                            nerf_length_min=-4.5, nerf_length_max=25.5, nerf_width_min=-12, nerf_width_max=12,nerf_height_min=-2, nerf_height_max=0.5,
                            pose_path=None,
                            subnerf_path=None,
                            re_loaddata=0,
                            result_path=None ):    
        super(maicity_dataload, self).__init__()
        self.split = split        
        self.cloud_size_val = cloud_size_val          #      
        self.result_path = result_path        
        self.re_loaddata = re_loaddata # 
        print("use maicity_dataload load train data")               
        if self.re_loaddata:        
            self.root_dir = root_dir
            self.data_start=data_start
            self.data_end=data_end        
            self.range_delete_x=range_delete_x      
            self.range_delete_y=range_delete_y    
            self.range_delete_z=range_delete_z      
            self.sub_nerf_test_num=sub_nerf_test_num #        
            self.subnerf_path=subnerf_path   
            self.surface_expand=surface_expand           
            print("self.surface_expand: ",self.surface_expand)
            self.nerf_width_min = nerf_width_min
            self.nerf_width_max = nerf_width_max        
            self.nerf_height_min = nerf_height_min
            self.nerf_height_max = nerf_height_max    
            self.nerf_length_min= nerf_length_min
            self.nerf_length_max= nerf_length_max       
            print("[",self.nerf_width_min,", ",self.nerf_width_max,"]")         
            print("[",self.nerf_length_min,", ",self.nerf_length_max,"]")                 
            print("[",self.nerf_height_min,", ",self.nerf_height_max,"]")                         

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
            self.positions=positions
            self.poses = torch.Tensor(poses)
        self.load_data()

    def load_data(self):
        if self.re_loaddata:
            sub_nerf_bound=np.zeros((self.sub_nerf_test_num,6)) 
            sub_nerf_bound_bigger=np.zeros((self.sub_nerf_test_num,6)) 
            sub_nerf_center_point=np.zeros((self.sub_nerf_test_num,3))            
            print("subnerf_path: ",self.subnerf_path)
            for i in range(self.sub_nerf_test_num):    
                file_name = f"{i+1}.pcd"
                file_path = os.path.join(self.subnerf_path, file_name)
                pcd = o3d.io.read_point_cloud(file_path) # 
                bbox = pcd.get_axis_aligned_bounding_box() # 
                min_bound = bbox.get_min_bound() # 
                max_bound = bbox.get_max_bound()
                extend_tmp0=0.025
                sub_nerf_bound[i][0]=min_bound[0]-extend_tmp0
                sub_nerf_bound[i][1]=min_bound[1]-extend_tmp0            
                sub_nerf_bound[i][2]=min_bound[2]-extend_tmp0
                sub_nerf_bound[i][3]=max_bound[0]+extend_tmp0           
                sub_nerf_bound[i][4]=max_bound[1]+extend_tmp0
                sub_nerf_bound[i][5]=max_bound[2]+extend_tmp0 
                
                extend_tmp=0.025
                sub_nerf_bound_bigger[i][0]=min_bound[0]-extend_tmp
                sub_nerf_bound_bigger[i][1]=min_bound[1]-extend_tmp            
                sub_nerf_bound_bigger[i][2]=min_bound[2]-extend_tmp
                sub_nerf_bound_bigger[i][3]=max_bound[0]+extend_tmp           
                sub_nerf_bound_bigger[i][4]=max_bound[1]+extend_tmp
                sub_nerf_bound_bigger[i][5]=max_bound[2]+extend_tmp             
                
                sub_nerf_center_point[i][0]=(min_bound[0]+max_bound[0])/2.0
                sub_nerf_center_point[i][1]=(min_bound[1]+max_bound[1])/2.0            
                sub_nerf_center_point[i][2]=(min_bound[2]+max_bound[2])/2.0                 
            np.set_printoptions(linewidth=np.inf)                   
            print("sub_nerf_bound:",sub_nerf_bound)               
            self.sub_nerf_num_count=np.zeros((self.sub_nerf_test_num,))# 

            self.rays_o = torch.ones((0,3))  # column 1~3
            self.rays_d = torch.ones((0,3))  # column 4~6
            self.parent_near_bound = torch.ones((0,1),dtype = torch.float)  # column 7               
            self.parent_far_bound = torch.ones((0,1),dtype = torch.float)      # column 8
            self.ray_class = torch.ones((0,1),dtype = torch.int)   # column 9          
            self.sub_nerf_num = torch.ones((0,1),dtype = torch.int) # column 10     
            self.near_bound = torch.ones((0,1),dtype = torch.float)    # column 11                 
            self.far_bound = torch.ones((0,1),dtype = torch.float)        # column 12             
            self.point_near_bound = torch.ones((0,1),dtype = torch.float)    # column 13               
            self.point_far_bound = torch.ones((0,1),dtype = torch.float)        # column 14           
            self.range_readings = torch.ones((0))            
        
        if self.re_loaddata:    
            for j in range(self.data_start, self.data_end):   

                file_path = os.path.join(self.root_dir, '{}.pcd'.format(j+1))          
                if((self.split == 'train') and (j+1-3-self.data_start)%5!=0):           # frame sparsity = 20%       
                # if((self.split == 'train') and (j+1-self.data_start)%4!=0):           # frame sparsity = 25%       
                # if((self.split == 'train') and (j+1-self.data_start)%3!=0):           # frame sparsity = 33%     
                # if((self.split == 'train') and (j+1-self.data_start)%2!=0):           # frame sparsity = 50%     
                # if((self.split == 'train') and (j+1-1-self.data_start)%3==0):      # frame sparsity = 67%      
                # if((self.split == 'train') and (j+1-1-self.data_start)%4==0):      # frame sparsity = 75%      
                # if((self.split == 'train') and (j+1-3-self.data_start)%5==0):      # frame sparsity = 80%      
                # if((self.split == 'train') and (j+1-5-self.data_start)%10==0):    # frame sparsity = 90%      
                    print("train file_path: ",file_path)
                elif (self.split == 'val' and (j+1-3-self.data_start)%5==0):      
                # elif (self.split == 'val' and 0 ):    # Note: More stringent conditions can be set to make the validation set empty                                                                 
                    print("val file_path: ",file_path)          
                else:                        
                    continue      
            
                points_source = pcl.PointCloud()    
                points_source = pcl.load(file_path)#
                print("source point cloud number: ",points_source.size)
                points_source_numpy = points_source.to_array()
                mask1 = np.logical_or.reduce((np.abs(points_source_numpy[:, 0]) >= self.range_delete_x, 
                                            np.abs(points_source_numpy[:, 1]) >= self.range_delete_y,
                                            np.abs(points_source_numpy[:, 2]) >= self.range_delete_z))
                points_effective2 = points_source_numpy[mask1]      # 
                dist = np.linalg.norm(points_effective2, axis=1) 
                mask3 = dist < 120 
                # mask3 = dist < 4                 
                points_effective3 = points_effective2[mask3]  # 
                ones_arr = np.ones((1, points_effective3.shape[0]))
                points_effective = np.vstack((points_effective3.T, ones_arr))        # 
                print("effective point cloud number: ",points_effective.shape)                    
                points_effective = self.poses[j] @ points_effective  #  
                points_effective = points_effective.T[:,:3]              
                
                mask4 = (points_effective[:, 0] >= self.nerf_length_min) & (points_effective[:, 1] >= self.nerf_width_min) & (points_effective[:, 2] >= self.nerf_height_min) & \
                    (points_effective[:, 0] <= self.nerf_length_max) & (points_effective[:, 1] <= self.nerf_width_max) & (points_effective[:, 2] <= self.nerf_height_max)
                points_effective = points_effective[mask4]  # 
                print("point cloud number in parent nerf aabb box: ",points_effective.shape)                           
                vec = points_effective - np.array([self.positions[j][0], self.positions[j][1], self.positions[j][2]]) # 
                dist_vec = np.linalg.norm(vec, axis=1) # 
                dir_vec = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, vec) #                     
                
                rays_d=np.zeros((points_effective.shape[0],3),dtype=float)    #            
                near_bound_parent=np.zeros((points_effective.shape[0],1),dtype=float)  #               
                far_bound_parent=np.zeros((points_effective.shape[0],1),dtype=float)
                ray_class=3*np.ones((points_effective.shape[0],1),dtype=int)                     
                sub_nerf_num_tmp = np.zeros((points_effective.shape[0],1),dtype=int)  #       
                            
                near_bound=np.zeros((points_effective.shape[0],1),dtype=float) # 
                far_bound=np.zeros((points_effective.shape[0],1),dtype=float)

                near_bound_point=np.zeros((points_effective.shape[0],1),dtype=float) # 
                far_bound_point=np.zeros((points_effective.shape[0],1),dtype=float)
                            
                range_readings=np.zeros((points_effective.shape[0],),dtype=float)  #      

                x_max=self.nerf_length_max # 
                x_min=self.nerf_length_min
                y_max=self.nerf_width_max
                y_min=self.nerf_width_min
                z_max=self.nerf_height_max
                z_min=self.nerf_height_min            
                
                count_sub_nerf=0 # 
                for i in range(0, points_effective.shape[0]):    
                    if 1:  #
                        is_inside, aabb_index = find_aabb_box(sub_nerf_center_point, sub_nerf_bound, points_effective[i])                
                        if is_inside:
                            # print("Query point is inside AABB box", aabb_index)
                            sub_nerf_num_tmp[count_sub_nerf]=aabb_index+1  
                            self.sub_nerf_num_count[aabb_index]=self.sub_nerf_num_count[aabb_index]+1 #  
                            in_sub_nerf=aabb_index #                                             
                        else:
                            continue
                            # print("Query point is not inside any AABB")                
                    
                    range_readings[count_sub_nerf]=dist_vec[i]
                    rays_d[count_sub_nerf]=dir_vec[i]

                    if 1: 
                        if 1 :  
                            near_bound[count_sub_nerf], far_bound[count_sub_nerf] = \
                                compute_far_bound0406(self.positions[j],rays_d[count_sub_nerf], sub_nerf_bound_bigger[in_sub_nerf][:3], sub_nerf_bound_bigger[in_sub_nerf][3:6])     
                            near_bound[count_sub_nerf]=near_bound[count_sub_nerf]-self.surface_expand 
                            far_bound[count_sub_nerf]=far_bound[count_sub_nerf]+self.surface_expand  
                            
                            near_bound_point[count_sub_nerf]=range_readings[count_sub_nerf]-self.surface_expand #  
                            far_bound_point[count_sub_nerf]=range_readings[count_sub_nerf]+self.surface_expand 
                            
                            far_bound_parent[count_sub_nerf]=compute_far_bound(self.positions[j],rays_d[count_sub_nerf], x_max, x_min, y_max, y_min, z_max, z_min)                           

                            if far_bound_parent[count_sub_nerf]<far_bound[count_sub_nerf]:  # 
                                far_bound_parent[count_sub_nerf]=far_bound[count_sub_nerf]
                                                                                    
                    count_sub_nerf=count_sub_nerf+1
                    
                print("point cloud number in child nerf aabb box: ",count_sub_nerf)                
                rays_d =rays_d[:count_sub_nerf]        
                near_bound_parent = near_bound_parent[:count_sub_nerf]                    
                far_bound_parent = far_bound_parent[:count_sub_nerf]                 
                ray_class =ray_class[:count_sub_nerf]      
                sub_nerf_num =sub_nerf_num_tmp[:count_sub_nerf]       
                near_bound =near_bound[:count_sub_nerf]          
                far_bound =far_bound[:count_sub_nerf]    
                
                near_bound_point =near_bound_point[:count_sub_nerf]          
                far_bound_point =far_bound_point[:count_sub_nerf]    
                
                range_readings =range_readings[:count_sub_nerf]                      
                    
                rays_o = np.broadcast_to(self.poses[j][:3,-1], np.shape(rays_d))      
                rays_o.flags.writeable = True
                rays_o = torch.Tensor(rays_o)            
                rays_d = torch.Tensor(rays_d)      
                near_bound_parent = torch.Tensor(near_bound_parent)           
                far_bound_parent = torch.Tensor(far_bound_parent)   
                ray_class = torch.Tensor(ray_class)           
                sub_nerf_num = torch.Tensor(sub_nerf_num)            
                            
                near_bound = torch.Tensor(near_bound)           
                far_bound = torch.Tensor(far_bound)                
                near_bound_point = torch.Tensor(near_bound_point)           
                far_bound_point = torch.Tensor(far_bound_point)       
            
                range_readings = torch.Tensor(range_readings)         
                print("near_bound_point[0]:",near_bound_point[0],"far_bound_point[0]:",far_bound_point[0])                     
                print("near_bound[0]:",near_bound[0],"far_bound[0]:",far_bound[0])         
                print("near_bound_parent[0]:",near_bound_parent[0],"far_bound_parent[0]:",far_bound_parent[0])         

                self.rays_o = torch.cat([self.rays_o, rays_o], 0)   
                self.rays_d = torch.cat([self.rays_d, rays_d], 0)   
                self.parent_near_bound = torch.cat([self.parent_near_bound, near_bound_parent], 0)  
                self.parent_far_bound = torch.cat([self.parent_far_bound, far_bound_parent], 0)             
                self.ray_class = torch.cat([self.ray_class, ray_class], 0)                    
                self.sub_nerf_num = torch.cat([self.sub_nerf_num, sub_nerf_num], 0)         
                self.range_readings = torch.cat([self.range_readings, range_readings], 0)  
                self.near_bound = torch.cat([self.near_bound, near_bound], 0)  
                self.far_bound = torch.cat([self.far_bound, far_bound], 0)                
                
                self.point_near_bound = torch.cat([self.point_near_bound, near_bound_point], 0)      
                self.point_far_bound = torch.cat([self.point_far_bound, far_bound], 0)             
                
                print("effective ray number: ",rays_o.shape[0])        

            self.rays = torch.cat([self.rays_o, self.rays_d,  
                                            self.parent_near_bound,self.parent_far_bound, 
                                            self.ray_class, self.sub_nerf_num, 
                                            self.near_bound,self.far_bound, 
                                            self.point_near_bound,self.point_far_bound,(self.range_readings).reshape(-1,1)], 
                                            dim=1)        
            self.ranges=self.range_readings      
            if((self.split == 'train')):      
                np.save(self.result_path+"/save_npy/split_child_nerf2_3/self_rays_train.npy",self.rays.data.cpu().numpy()) # 
                np.save(self.result_path+"/save_npy/split_child_nerf2_3/self_ranges_train.npy",self.ranges.data.cpu().numpy()) #       
                print("Save file to",self.result_path+"/save_npy/split_child_nerf2_3/self_rays_train.npy")  
                print("Save file to",self.result_path+"/save_npy/split_child_nerf2_3/self_ranges_train.npy")              
            if((self.split == 'val')):       
                np.save(self.result_path+"/save_npy/split_child_nerf2_3/self_rays_val.npy",self.rays.data.cpu().numpy()) # 
                np.save(self.result_path+"/save_npy/split_child_nerf2_3/self_ranges_val.npy",self.ranges.data.cpu().numpy()) #       
                print("Save file to",self.result_path+"/save_npy/split_child_nerf2_3/self_rays_val.npy")  
                print("Save file to",self.result_path+"/save_npy/split_child_nerf2_3/self_ranges_val.npy")       
        else: #  
            if((self.split == 'train')):             
                dataset_ranges=np.load(self.result_path+'/save_npy/split_child_nerf2_3/self_ranges_train.npy')  #
                print(self.result_path+'/save_npy/split_child_nerf2_3/self_ranges_train.npy')
                self.ranges = torch.tensor(dataset_ranges)
                dataset_rays=np.load(self.result_path+'/save_npy/split_child_nerf2_3/self_rays_train.npy')  #              
                self.rays = torch.tensor(dataset_rays)                
            if((self.split == 'val')):       
                dataset_ranges=np.load(self.result_path+'/save_npy/split_child_nerf2_3/self_ranges_val.npy')  #
                self.ranges = torch.tensor(dataset_ranges)
                dataset_rays=np.load(self.result_path+'/save_npy/split_child_nerf2_3/self_rays_val.npy')  #                           
                self.rays = torch.tensor(dataset_rays)     

        print("======================================")        
        print("ray number:",self.rays.shape)       
                
        if self.split == 'train':
            self.all_rays=self.rays
            self.all_ranges=self.ranges
            print("======================================")
            print("train ray number:",self.all_rays.shape)
            print("======================================")

    def __getitem__(self, index):
        if self.split == 'train':
            sample = {'rays': self.all_rays[index],
                      'ranges': self.all_ranges[index]}

        else:
            rays_val=np.zeros((self.cloud_size_val,self.rays.shape[1]),dtype=float)    
            ranges_val=np.zeros((self.cloud_size_val,),dtype=float)                  
            select=torch.linspace(1, self.rays.shape[0]-2, steps=self.cloud_size_val, dtype=torch.float32) 
            for i in range(0, len(select)):
                rays_val[i]=self.rays[math.floor(select[i])]
                ranges_val[i]=self.ranges[math.floor(select[i])]
            rays_val = torch.Tensor(rays_val)        
            ranges_val = torch.Tensor(ranges_val)        
            # print("======================================")
            # print("val ray number:",rays_val.shape)
            # print("======================================")

            sample = {'rays': rays_val[index],
                      'ranges': ranges_val[index]}
           
        return sample

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        else:
            return self.cloud_size_val


class kitti_dataload(data.Dataset):
    def __init__(self, root_dir, split='train', data_start=1439,data_end=1510,cloud_size_val=2048,
                            range_delete_x=2, range_delete_y=1, range_delete_z=0.5,
                            sub_nerf_test_num=3,
                            surface_expand=0.1,   
                            over_height=0.168, over_low=-2, 
                            interest_x=12, 
                            interest_y=12,               
                            pose_path= None,
                            subnerf_path= None,
                            parentnerf_path=None,
                            re_loaddata=0,
                            result_path=None,                            
                            ):    
        super(kitti_dataload, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.cloud_size_val = cloud_size_val       
        self.parentnerf_path = parentnerf_path                 
        self.result_path = result_path                 
        self.re_loaddata = re_loaddata      
        print("use kitti_dataload load train data")                     
        if self.re_loaddata:
            self.data_start=data_start
            self.data_end=data_end        
            self.range_delete_x=range_delete_x      
            self.range_delete_y=range_delete_y    
            self.range_delete_z=range_delete_z      
            self.sub_nerf_test_num=sub_nerf_test_num #        
            self.subnerf_path=subnerf_path   
            self.surface_expand=surface_expand       
            self.over_height=over_height    
            self.over_low=over_low            
            self.interest_x=interest_x   
            self.interest_y=interest_y        

            parent_nerf_pcd = o3d.io.read_point_cloud(parentnerf_path)
            parent_nerf_bbox = parent_nerf_pcd.get_axis_aligned_bounding_box()
            parent_nerf_min_bound = parent_nerf_bbox.get_min_bound()
            parent_nerf_max_bound = parent_nerf_bbox.get_max_bound()        
            self.nerf_length_min= parent_nerf_min_bound[0]
            self.nerf_width_min = parent_nerf_min_bound[1]        
            self.nerf_height_min = parent_nerf_min_bound[2]        
            self.nerf_length_max= parent_nerf_max_bound[0]              
            self.nerf_width_max = parent_nerf_max_bound[1]        
            self.nerf_height_max = parent_nerf_max_bound[2]             
            print("[",self.nerf_length_min,", ",self.nerf_length_max,"]")            
            print("[",self.nerf_width_min,", ",self.nerf_width_max,"]")                      
            print("[",self.nerf_height_min,", ",self.nerf_height_max,"]")                         

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
            
            T_start=poses[data_start+1] #          
            T_start_inv = np.linalg.inv(T_start)
            T_start_inv = torch.from_numpy(T_start_inv).float()        
            poses = torch.Tensor(poses)    
            poses_start= T_start_inv @ poses #    
            # self.poses = torch.Tensor(poses)
            self.poses = poses_start      
            positions=self.poses[: ,:3,-1]
            self.positions=positions
        
        self.load_data()

    def load_data(self):
        if self.re_loaddata:                    
            sub_nerf_bound=np.zeros((self.sub_nerf_test_num,6)) 
            sub_nerf_bound_bigger=np.zeros((self.sub_nerf_test_num,6)) 
            sub_nerf_center_point=np.zeros((self.sub_nerf_test_num,3))            
            print("subnerf_path:",self.subnerf_path)         
            for i in range(self.sub_nerf_test_num):    
                file_name = f"{i+1}.pcd"
                file_path = os.path.join(self.subnerf_path, file_name)
                pcd = o3d.io.read_point_cloud(file_path) # 
                bbox = pcd.get_axis_aligned_bounding_box() # 
                min_bound = bbox.get_min_bound() # 
                max_bound = bbox.get_max_bound()
                extend_tmp=0.025
                sub_nerf_bound[i][0]=min_bound[0]-extend_tmp
                sub_nerf_bound[i][1]=min_bound[1]-extend_tmp            
                sub_nerf_bound[i][2]=min_bound[2]-extend_tmp
                sub_nerf_bound[i][3]=max_bound[0]+extend_tmp           
                sub_nerf_bound[i][4]=max_bound[1]+extend_tmp
                sub_nerf_bound[i][5]=max_bound[2] +extend_tmp
                
                extend_tmp2=0.025 # 
                sub_nerf_bound_bigger[i][0]=min_bound[0]-extend_tmp2
                sub_nerf_bound_bigger[i][1]=min_bound[1]-extend_tmp2            
                sub_nerf_bound_bigger[i][2]=min_bound[2]-extend_tmp2
                sub_nerf_bound_bigger[i][3]=max_bound[0]+extend_tmp2           
                sub_nerf_bound_bigger[i][4]=max_bound[1]+extend_tmp2
                sub_nerf_bound_bigger[i][5]=max_bound[2]+extend_tmp2             
                
                sub_nerf_center_point[i][0]=(min_bound[0]+max_bound[0])/2.0
                sub_nerf_center_point[i][1]=(min_bound[1]+max_bound[1])/2.0            
                sub_nerf_center_point[i][2]=(min_bound[2]+max_bound[2])/2.0                 
            np.set_printoptions(linewidth=np.inf)                   
            print("sub_nerf_bound.shape:",sub_nerf_bound.shape)                 
            self.sub_nerf_num_count=np.zeros((self.sub_nerf_test_num,))# 

            self.rays_o = torch.ones((0,3))  # column 1~3
            self.rays_d = torch.ones((0,3))  # column 4~6
            self.parent_near_bound = torch.ones((0,1),dtype = torch.float)  # column 7                
            self.parent_far_bound = torch.ones((0,1),dtype = torch.float)      # column 8             
            self.ray_class = torch.ones((0,1),dtype = torch.int)   # column 9                 
            self.sub_nerf_num = torch.ones((0,1),dtype = torch.int) # column 10           
            self.near_bound = torch.ones((0,1),dtype = torch.float)    # column11                     
            self.far_bound = torch.ones((0,1),dtype = torch.float)        # column 12                  
            self.point_near_bound = torch.ones((0,1),dtype = torch.float)    # column 13                    
            self.point_far_bound = torch.ones((0,1),dtype = torch.float)        # column14               
            self.range_readings = torch.ones((0))   

        if self.re_loaddata:
            for j in range(self.data_start, self.data_end):    

                file_path = os.path.join(self.root_dir, '{}.pcd'.format(j+1))          
                if((self.split == 'train') and (j+1-3-self.data_start)%5!=0):          # frame sparsity = 20%   
                # if((self.split == 'train') and (j+1-self.data_start)%4!=0):           # frame sparsity = 25%       
                # if((self.split == 'train') and (j+1-self.data_start)%3!=0):           # frame sparsity = 33%     
                # if((self.split == 'train') and (j+1-self.data_start)%2!=0):           # frame sparsity = 50%     
                # if((self.split == 'train') and (j+1-1-self.data_start)%3==0):      # frame sparsity = 67%      
                # if((self.split == 'train') and (j+1-1-self.data_start)%4==0):      # frame sparsity = 75%      
                # if((self.split == 'train') and (j+1-3-self.data_start)%5==0):      # frame sparsity = 80%      
                # if((self.split == 'train') and (j+1-5-self.data_start)%10==0):    # frame sparsity = 90%                          
                    print("train file_path: ",file_path)
                elif (self.split == 'val' and (j+1-3)%5==0):             
                # elif (self.split == 'val' and 0 ):    # Note: More stringent conditions can be set to make the validation set empty      
                    print("val file_path: ",file_path)          
                else:                        
                    continue      
                
                points_source = pcl.PointCloud()    
                points_source = pcl.load(file_path)#
                print("source point cloud number: ",points_source.size)
                points_source_numpy = points_source.to_array()
                mask1 = np.logical_or.reduce((np.abs(points_source_numpy[:, 0]) >= self.range_delete_x, 
                                            np.abs(points_source_numpy[:, 1]) >= self.range_delete_y,
                                            np.abs(points_source_numpy[:, 2]) >= self.range_delete_z))
                points_effective2 = points_source_numpy[mask1]      # 
                dist = np.linalg.norm(points_effective2, axis=1) 
                mask3 = dist <= 120 
                # mask3 = dist <= 4.5                         
                points_effective3 = points_effective2[mask3]  # 
                
                mask_tmp = points_effective3[:, 2] <= self.over_height
                points_effective3 = points_effective3[mask_tmp]  #            
                mask_tmp = points_effective3[:, 2] >= self.over_low
                points_effective3 = points_effective3[mask_tmp]  #        

                ones_arr = np.ones((1, points_effective3.shape[0]))
                points_effective = np.vstack((points_effective3.T, ones_arr))        # 
                print("effective point cloud number: ",points_effective.shape)                    
                points_effective = self.poses[j+1] @ points_effective  # 
                points_effective = points_effective.T[:,:3]              
                print("points_effective.shape[0]:",points_effective.shape[0])
                
                count_nerf=0 # 
                points_nerf_tmp=np.zeros((points_effective.shape[0], 3),dtype=float)       
                print("self.interest_x:",self.interest_x,"self.interest_y:",self.interest_y)               
                for i in range(0, points_effective.shape[0]):        
                    near_all_pose=False # 
                    for k in range(self.data_start, self.data_end):
                        if(abs(points_effective[i][0]-self.poses[k+1][0,-1])>self.interest_x or abs(points_effective[i][1]-self.poses[k+1][1,-1])>self.interest_y ):           #                                                   
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
                print("point cloud number in nerf: ",count_nerf)                
                points_effective=points_nerf_tmp[:count_nerf]                          

                vec = points_effective - np.array([self.positions[j+1][0], self.positions[j+1][1], self.positions[j+1][2]]) 

                dist_vec = np.linalg.norm(vec, axis=1) # 
                dir_vec = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, vec) #                     
                
                rays_d=np.zeros((points_effective.shape[0],3),dtype=float)    #            
                near_bound_parent=np.zeros((points_effective.shape[0],1),dtype=float)#          
                far_bound_parent=np.zeros((points_effective.shape[0],1),dtype=float)
                ray_class=3*np.ones((points_effective.shape[0],1),dtype=int)                    
                sub_nerf_num_tmp = np.zeros((points_effective.shape[0],1),dtype=int)  #       
                            
                near_bound=np.zeros((points_effective.shape[0],1),dtype=float) #     
                far_bound=np.zeros((points_effective.shape[0],1),dtype=float)

                near_bound_point=np.zeros((points_effective.shape[0],1),dtype=float) #     
                far_bound_point=np.zeros((points_effective.shape[0],1),dtype=float)
                            
                range_readings=np.zeros((points_effective.shape[0],),dtype=float)  #      

                x_max=self.nerf_length_max # 
                x_min=self.nerf_length_min
                y_max=self.nerf_width_max
                y_min=self.nerf_width_min
                z_max=self.nerf_height_max
                z_min=self.nerf_height_min            

                count_sub_nerf0=0 #             
                count_sub_nerf=0 # 
                for i in range(0, points_effective.shape[0]):    
                    if 1:  
                        is_inside, aabb_index = find_aabb_box(sub_nerf_center_point, sub_nerf_bound, points_effective[i])            #      
                        if is_inside:
                            sub_nerf_num_tmp[count_sub_nerf]=aabb_index+1  #              
                            self.sub_nerf_num_count[aabb_index]=self.sub_nerf_num_count[aabb_index]+1 #     
                            in_sub_nerf=aabb_index #                                             
                        else:
                            continue
                    count_sub_nerf0=count_sub_nerf0+1
                    
                    range_readings[count_sub_nerf]=dist_vec[i]
                    rays_d[count_sub_nerf]=dir_vec[i]

                    if 1: 
                        if 1 :  
                            intersect, near_bound[count_sub_nerf], far_bound[count_sub_nerf] = \
                                compute_far_bound0606(self.positions[j+1],rays_d[count_sub_nerf], sub_nerf_bound_bigger[in_sub_nerf][:3], sub_nerf_bound_bigger[in_sub_nerf][3:6])     #  

                            if  intersect == False:
                                continue                     
                            near_bound[count_sub_nerf]=near_bound[count_sub_nerf]-self.surface_expand 
                            far_bound[count_sub_nerf]=far_bound[count_sub_nerf]+self.surface_expand  
                            
                            near_bound_point[count_sub_nerf]=range_readings[count_sub_nerf]-self.surface_expand #  
                            far_bound_point[count_sub_nerf]=range_readings[count_sub_nerf]+self.surface_expand 
                            
                            far_bound_parent[count_sub_nerf]=compute_far_bound(self.positions[j+1],rays_d[count_sub_nerf], x_max, x_min, y_max, y_min, z_max, z_min)     #                              
                            
                            if far_bound_parent[count_sub_nerf]<far_bound[count_sub_nerf]: 
                                far_bound_parent[count_sub_nerf]=far_bound[count_sub_nerf]

                    count_sub_nerf=count_sub_nerf+1

                print("count_sub_nerf0: ",count_sub_nerf0)                    
                print("count_sub_nerf: ",count_sub_nerf)                
                rays_d =rays_d[:count_sub_nerf]        
                near_bound_parent = near_bound_parent[:count_sub_nerf]                    
                far_bound_parent = far_bound_parent[:count_sub_nerf]                 
                ray_class =ray_class[:count_sub_nerf]      
                sub_nerf_num =sub_nerf_num_tmp[:count_sub_nerf]       
                near_bound =near_bound[:count_sub_nerf]          
                far_bound =far_bound[:count_sub_nerf]    
                
                near_bound_point =near_bound_point[:count_sub_nerf]          
                far_bound_point =far_bound_point[:count_sub_nerf]    
                
                range_readings =range_readings[:count_sub_nerf]                      
                    
                rays_o = np.broadcast_to(self.poses[j+1][:3,-1], np.shape(rays_d))            #            
                rays_o.flags.writeable = True
                rays_o = torch.Tensor(rays_o)            
                rays_d = torch.Tensor(rays_d)      
                near_bound_parent = torch.Tensor(near_bound_parent)           
                far_bound_parent = torch.Tensor(far_bound_parent)   
                ray_class = torch.Tensor(ray_class)           
                sub_nerf_num = torch.Tensor(sub_nerf_num)            
                            
                near_bound = torch.Tensor(near_bound)           
                far_bound = torch.Tensor(far_bound)                
                near_bound_point = torch.Tensor(near_bound_point)           
                far_bound_point = torch.Tensor(far_bound_point)       
            
                range_readings = torch.Tensor(range_readings)         
                print("near_bound_point[0]:",near_bound_point[0],"far_bound_point[0]:",far_bound_point[0])                     
                print("near_bound[0]:",near_bound[0],"far_bound[0]:",far_bound[0])         
                print("near_bound_parent[0]:",near_bound_parent[0],"far_bound_parent[0]:",far_bound_parent[0])         

                self.rays_o = torch.cat([self.rays_o, rays_o], 0)   
                self.rays_d = torch.cat([self.rays_d, rays_d], 0)  
                self.parent_near_bound = torch.cat([self.parent_near_bound, near_bound_parent], 0)   
                self.parent_far_bound = torch.cat([self.parent_far_bound, far_bound_parent], 0)               
                self.ray_class = torch.cat([self.ray_class, ray_class], 0)                   
                self.sub_nerf_num = torch.cat([self.sub_nerf_num, sub_nerf_num], 0)     #            
                self.range_readings = torch.cat([self.range_readings, range_readings], 0)  # 
                self.near_bound = torch.cat([self.near_bound, near_bound], 0)        
                self.far_bound = torch.cat([self.far_bound, far_bound], 0)             
                
                self.point_near_bound = torch.cat([self.point_near_bound, near_bound_point], 0)  
                self.point_far_bound = torch.cat([self.point_far_bound, far_bound], 0)             
                
                print("effective ray number: ",rays_o.shape[0])        

            self.rays = torch.cat([self.rays_o, self.rays_d, 
                                            self.parent_near_bound,self.parent_far_bound, 
                                            self.ray_class, self.sub_nerf_num, 
                                            self.near_bound,self.far_bound, 
                                            self.point_near_bound,self.point_far_bound,(self.range_readings).reshape(-1,1)], 
                                            dim=1)        
            self.ranges=self.range_readings    

            if((self.split == 'train')):      
                np.save(self.result_path+"/save_npy/split_child_nerf2_3/self_rays_train.npy",self.rays.data.cpu().numpy()) # 
                np.save(self.result_path+"/save_npy/split_child_nerf2_3/self_ranges_train.npy",self.ranges.data.cpu().numpy()) #       
                print("Save file to",self.result_path+"/save_npy/split_child_nerf2_3/self_rays_train.npy")  
                print("Save file to",self.result_path+"/save_npy/split_child_nerf2_3/self_ranges_train.npy")              
            if((self.split == 'val')):       
                np.save(self.result_path+"/save_npy/split_child_nerf2_3/self_rays_val.npy",self.rays.data.cpu().numpy()) # 
                np.save(self.result_path+"/save_npy/split_child_nerf2_3/self_ranges_val.npy",self.ranges.data.cpu().numpy()) #       
                print("Save file to",self.result_path+"/save_npy/split_child_nerf2_3/self_rays_val.npy")  
                print("Save file to",self.result_path+"/save_npy/split_child_nerf2_3/self_ranges_val.npy")       
        else: #  
            if((self.split == 'train')):             
                dataset_ranges=np.load(self.result_path+'/save_npy/split_child_nerf2_3/self_ranges_train.npy')  #
                print(self.result_path+'/save_npy/split_child_nerf2_3/self_ranges_train.npy')
                self.ranges = torch.tensor(dataset_ranges)
                dataset_rays=np.load(self.result_path+'/save_npy/split_child_nerf2_3/self_rays_train.npy')  #              
                self.rays = torch.tensor(dataset_rays)                
            if((self.split == 'val')):       
                dataset_ranges=np.load(self.result_path+'/save_npy/split_child_nerf2_3/self_ranges_val.npy')  #
                self.ranges = torch.tensor(dataset_ranges)
                dataset_rays=np.load(self.result_path+'/save_npy/split_child_nerf2_3/self_rays_val.npy')  #                           
                self.rays = torch.tensor(dataset_rays)     

        print("======================================")        
        print("ray number:",self.rays.shape)       
                
        if self.split == 'train':
            self.all_rays=self.rays
            self.all_ranges=self.ranges
            print("======================================")
            print("train ray total number:",self.all_rays.shape)
            print("======================================")

    def __getitem__(self, index):
        if self.split == 'train':
            sample = {'rays': self.all_rays[index],
                      'ranges': self.all_ranges[index]}

        else:
            rays_val=np.zeros((self.cloud_size_val,self.rays.shape[1]),dtype=float)    
            ranges_val=np.zeros((self.cloud_size_val,),dtype=float)                  
            select=torch.linspace(1, self.rays.shape[0]-2, steps=self.cloud_size_val, dtype=torch.float32) 
            for i in range(0, len(select)):
                rays_val[i]=self.rays[math.floor(select[i])]
                ranges_val[i]=self.ranges[math.floor(select[i])]
            rays_val = torch.Tensor(rays_val)        
            ranges_val = torch.Tensor(ranges_val)        
            # print("======================================")
            # print("val ray total number:",rays_val.shape)
            # print("======================================")

            sample = {'rays': rays_val[index],
                      'ranges': ranges_val[index]}
           
        return sample

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        else:
            return self.cloud_size_val




