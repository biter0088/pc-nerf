#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import numpy as np
import open3d as o3d
import pcl
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import random
from scipy.spatial.distance import cdist

def region_growing_segmentation(pcd, radius, min_cluster_size, max_cluster_size,
                                folder_name=None):
    # Convert Open3D.PointCloud to numpy array
    xyz = np.asarray(pcd.points)

    # KDTree for nearest neighbor search
    tree = o3d.geometry.KDTreeFlann(pcd)

    # Create a bool array to keep track of visited points
    visited = np.zeros(xyz.shape[0], dtype=bool)

    # Create a list to store the point cloud clusters
    clusters = []
    unclusters=np.zeros((0,3))#
    # Iterate over all points in the point cloud
    for i in tqdm(range(xyz.shape[0])):
        # If the point has not been visited yet, start a new cluster
        if not visited[i]:
            visited[i] = True
            cluster = [i]

            current_index_count=0 #
            # Loop over all points in the cluster
            while len(cluster) > 0:
                # Get the last point added to the cluster
                current_index = cluster[current_index_count]                

                # Find all neighboring points within a radius
                [k, idx, _] = tree.search_radius_vector_3d(xyz[current_index], radius)

                # Check if the number of neighboring points is within the desired range
                # if min_cluster_size <= len(idx) <= max_cluster_size: # 
                # Add the neighboring points to the cluster
                for j in idx:
                    if not visited[j]:
                        visited[j] = True
                        cluster.append(j)
                            
                current_index_count = current_index_count+1# 
                # 
                if current_index_count>=len(cluster) :
                    break                                            

            # Add the cluster to the list of clusters
            # clusters.append(pcd.select_by_index(cluster))            
            if min_cluster_size <= len(cluster) <= max_cluster_size: #  
                clusters.append(pcd.select_by_index(cluster))
            else:
                xyz_tmp = np.asarray(pcd.select_by_index(cluster).points)
                unclusters=np.concatenate((unclusters,xyz_tmp),axis=0)
                
    print("uncluster point number:",unclusters.shape[0])    
    pcd_unclusters = o3d.geometry.PointCloud()
    pcd_unclusters.points = o3d.utility.Vector3dVector(unclusters)                

    pcd_unclusters_file_name = f"unclusters.pcd"
    pcd_unclusters_file_path = os.path.join(folder_name, pcd_unclusters_file_name)    
    print("pcd_unclusters_file_path:",pcd_unclusters_file_path)
    o3d.io.write_point_cloud(pcd_unclusters_file_path, pcd_unclusters)      
    # print("clusters:",clusters)
    
    for i in range(len(clusters)):    
        file_name = f"point_cloud_{i}.pcd"
        file_path = os.path.join(folder_name, file_name)
        o3d.io.write_point_cloud(file_path, clusters[i])         
    
    # Create a list to store the AABB bounding boxes
    aabb_list = []
    # Loop over all clusters and compute the AABB bounding box
    for cluster in clusters:
        aabb = cluster.get_axis_aligned_bounding_box()
        color = np.random.rand(3)        
        aabb.color = (color[0], color[1], color[2])  # Set AABB color to red
        aabb_list.append(aabb)

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add all clusters and AABB bounding boxes to the visualization window
    print("len(clusters):",len(clusters))
    print("len(aabb_list):",len(aabb_list))    

    success_clusters=np.zeros((0,3))    
    for i in range(len(clusters)):
        xyz_tmp = np.asarray(clusters[i].points)
        success_clusters=np.concatenate((success_clusters,xyz_tmp),axis=0)    
        vis.add_geometry(clusters[i])
        vis.add_geometry(aabb_list[i])
        # vis.run()# 
    print("sucess clusters number:",len(clusters))    
    print("sucess clusters point number:",success_clusters.shape[0])    
                    
    pcd_success_clusters = o3d.geometry.PointCloud()
    pcd_success_clusters.points = o3d.utility.Vector3dVector(success_clusters)                
    pcd_success_clusters_file_name = f"success_clusters.pcd"
    pcd_success_clusters_file_path = os.path.join(folder_name, pcd_success_clusters_file_name)    
    print("pcd_success_clusters_file_path:",pcd_success_clusters_file_path)    
    o3d.io.write_point_cloud(pcd_success_clusters_file_path, pcd_success_clusters)      
            
    # Run the visualization
    vis.run() #

    # Return the list of point cloud clusters
    return clusters, aabb_list


if __name__ == '__main__':     
    
    pcd_path = "data_preprocess/kitti_pre_processed/sequence00/1151_1200_view/sub_pointcloud/points_no_ground.pcd"        
    pcd = o3d.io.read_point_cloud(pcd_path)        
    points = np.asarray(pcd.points)
    points_num=points.shape[0]        
    folder_name = f"data_preprocess/kitti_pre_processed/sequence00/1151_1200_view/sub_pointcloud/no_ground_clusters/"            
    clusters,aabb_list = region_growing_segmentation(pcd, radius=0.35, min_cluster_size=15,max_cluster_size=points_num, folder_name=folder_name)