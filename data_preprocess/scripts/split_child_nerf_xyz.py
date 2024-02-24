import open3d as o3d
import numpy as np
import sys
import os

def huafen(length, xy_threshold, x_min, x_max):
    if length > 2 * xy_threshold:
        if( length % xy_threshold <= 0.5 * xy_threshold):
            x_split_num =  int(length / xy_threshold) # 
        else:
            x_split_num =  int(length / xy_threshold) + 1  #  
        x_split_num = x_split_num + 1 #  1
    else:
        x_split_num =2 # 
    x_splits = np.zeros((x_split_num), dtype=float)    #     
    for i in range(x_split_num):
        x_splits[i] = x_min + i * xy_threshold

    x_splits[-1] =  x_max+0.05 # 
    return x_splits

def split_pointcloud2(pcd_np, xy_threshold, z_threshold, save_dir,source_points_num,child_id):
    print_flag=0
    split_pcd_point_num = 0
    id = 0
    aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pcd_np))
    x_min, x_max = aabb.get_min_bound()[0], aabb.get_max_bound()[0]    
    y_min, y_max = aabb.get_min_bound()[1], aabb.get_max_bound()[1]    
    z_min, z_max = aabb.get_min_bound()[2], aabb.get_max_bound()[2]   
    length = x_max-x_min      
    width = y_max-y_min    
    height = z_max-z_min
    x_splits = huafen(length, xy_threshold, x_min, x_max)           
    y_splits = huafen(width, xy_threshold, y_min, y_max)       
    z_splits = huafen(height, z_threshold, z_min, z_max)  

    for k in range(z_splits.shape[0]-1):   
        for j in range(y_splits.shape[0]-1):                 
            for i in range(x_splits.shape[0]-1): # 
                mask = (pcd_np[:, 0] >= x_splits[i]) & (pcd_np[:, 0] < x_splits[i+1]) &\
                                (pcd_np[:, 1] >= y_splits[j]) & (pcd_np[:, 1] < y_splits[j+1]) & \
                                (pcd_np[:, 2] >= z_splits[k]) & (pcd_np[:, 2] < z_splits[k+1])     
                points_id = pcd_np[mask]        
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_id)
                if((np.asarray(pcd.points)).shape[0] > 0 ):
                    o3d.io.write_point_cloud(f"{save_dir}/{child_id}_{id}.pcd", pcd)                    
                    id=id+1     
                    split_pcd_point_num=split_pcd_point_num+points_id.shape[0]      
                    if(print_flag):
                        print(child_id,"_",id,".pcd")                                                              

    if(source_points_num!=split_pcd_point_num):
        print("not complete")  
        # print("source_points_num:",source_points_num)
        # print("split_pcd_point_num:",split_pcd_point_num)        
           
if __name__ == '__main__':
    subnerf_path="data_preprocess/kitti_pre_processed/sequence00/1151_1200_view/sub_pointcloud/child_nerf/"
    sub_nerf_num=91     
    output_path_prefix="data_preprocess/kitti_pre_processed/sequence00/1151_1200_view/sub_pointcloud/child_nerf2/"   
    for i in range(sub_nerf_num):    
        file_name = f"{i+1}.pcd"
        pcd_file = os.path.join(subnerf_path, file_name)
        print("child nerf:",file_name,"divide into================")           

        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd_np = np.asarray(pcd.points)
        points_num=pcd_np.shape[0]    
        xy_threshold=1  # can be adjust
        z_threshold=1    # can be adjust
        split_pointcloud2(pcd_np, xy_threshold,z_threshold, output_path_prefix,points_num,i+1)        

