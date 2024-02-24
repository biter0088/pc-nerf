import os
import open3d as o3d

def load_and_display_pointclouds(folder_path):
    """
    Load all pcd files in a folder, and display them and their AABB bounding boxes.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing pcd files
    """
    # Load all pcd files in the folder
    pcd_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pcd")]
    pointclouds = [o3d.io.read_point_cloud(f) for f in pcd_files]
    
    # Create a list of geometries that includes the point clouds and their AABB bounding boxes
    geometries = pointclouds.copy()
    for pcd in pointclouds:
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
        aabb.color = [1, 0, 0]
        geometries.append(aabb)
        
    # Visualize all geometries in one window
    o3d.visualization.draw_geometries(geometries)

# load_and_display_pointclouds('./')
load_and_display_pointclouds('data_preprocess/kitti_pre_processed/sequence00/1151_1200_view/sub_pointcloud/child_nerf/')
