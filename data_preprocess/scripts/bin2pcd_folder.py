# refer to semantic kitti dataset
import numpy as np
import struct
import os
import sys
import open3d as o3d

def bin_to_pcd(binFileName, labelFileName):
    label = np.fromfile(labelFileName, dtype=np.uint32)#load label
    size_float = 4
    list_pcd = []
    pcd_count=0
    with open(binFileName, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            # refer to semantic-kitti-api/config/semantic-kitti.yaml   [https://github.com/PRBonn/semantic-kitti-api]  
            if(((label[pcd_count]>=252) and (label[pcd_count]<=259))  # remove dynamic objects       
                  or (label[pcd_count]==0) # unlabeled                          
                  or (label[pcd_count]==1) # outlier                     
                  or (label[pcd_count]==10) # car
                  or (label[pcd_count]==11) # bicycle
                  or (label[pcd_count]==13) # bus
                  or (label[pcd_count]==15) # motorcycle
                  or (label[pcd_count]==16) # on-rails
                  or (label[pcd_count]==18) # truck
                  or (label[pcd_count]==20) # other-vehicle
                  or (label[pcd_count]==30) # person
                  or (label[pcd_count]==31) # bicyclist                         
                  or (label[pcd_count]==32) # motorcyclist           
                  or (label[pcd_count]==99) # other-object                  
                  or (label[pcd_count]==251) # moving  lidar-mos mod moving                     
               ):     
                byte = f.read(size_float * 4)
                pcd_count=pcd_count+1         
                continue
   
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
            pcd_count=pcd_count+1

    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    print("len(pcd.points):",len(pcd.points))#add hxz    
    print("label.shape[0]:",label.shape[0])
    return pcd

def main(binFolderName, labelFolderName, pcdFolderName):
    for i in os.listdir(binFolderName):
        binFileName = binFolderName+i
        print(i)
        labelFileName = labelFolderName+i[:-4]+'.label'
        print(i[:-4]+'.label')

        pcd = bin_to_pcd(binFileName, labelFileName)
        pcdFileName = pcdFolderName+i[:-4]+'.pcd'
        print(pcdFileName)
        o3d.io.write_point_cloud(pcdFileName, pcd)
        print("==============================")

if __name__ == "__main__":
    a = sys.argv[1]
    b = sys.argv[2]
    c = sys.argv[3]
    main(a, b, c)