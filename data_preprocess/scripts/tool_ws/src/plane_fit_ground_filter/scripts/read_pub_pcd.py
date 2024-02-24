#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import numpy as np
import open3d as o3d

def read_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    # print(np.asarray(pcd.points))
    colors = np.asarray(pcd.colors) * 255
    points = np.asarray(pcd.points)
    print(points.shape, colors.shape)
# 	return np.concatenate([points, colors], axis=-1)
    return points

def select_interest(points_source):
    points_effective_tmp=np.zeros((points_source.shape[0],3),dtype=float)
    count_effective=0
    for i in range(0, points_source.shape[0]):     #
        # if(points_source[i][2]<-1.5):#    
        #     continue
        # if(points_source[i][2]>-0.2):#    
        #     continue    

        # if(abs(points_source[i][1]) > 10):
        #     continue                
        # if(abs(points_source[i][0]) > 20):
            # continue               
        # points_effective_tmp[count_effective][0]=points_source[i][0]+9                

        points_effective_tmp[count_effective][0]=points_source[i][0]                       
        points_effective_tmp[count_effective][1]=points_source[i][1]         
        points_effective_tmp[count_effective][2]=points_source[i][2]  
        count_effective=count_effective+1
    print("count: ",count_effective)                
    points_effective=points_effective_tmp[:count_effective]              
    print("points_effective.shape:",points_effective.shape)                
    return  points_effective            

def talker():

    pub = rospy.Publisher('source', PointCloud2, queue_size=5)
    rospy.init_node('pointcloud_publisher_node', anonymous=True)
    rate = rospy.Rate(1)

    pcd_path = rospy.get_param("pcd_path")
    point_pcd=read_pcd(pcd_path)          
    points = select_interest(point_pcd)

    while not rospy.is_shutdown():

        msg = PointCloud2()
        msg.header.stamp = rospy.Time().now()
        msg.header.frame_id = "map"

        if len(points.shape) == 3:
            msg.height = points.shape[1]
            msg.width = points.shape[0]
        else:
            msg.height = 1
            msg.width = len(points)

        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = False
        msg.data = np.asarray(points, np.float32).tobytes()        
        pub.publish(msg)
        # print("published...")
        rate.sleep()

if __name__ == '__main__':     
    talker()