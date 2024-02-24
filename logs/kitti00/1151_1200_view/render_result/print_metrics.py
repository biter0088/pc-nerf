import numpy as np
import open3d as o3d
import os
import sys

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
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances

# def eval_pts(pts1, pts2, threshold=0.5):
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
    return np.mean(value)#

def acc_thres(pred, gt,threshold=0.2):
    error = np.abs(pred - gt)
    # acc_thres = error < 0.2 
    acc_thres = error < threshold     
    acc_thres = np.sum(acc_thres) / acc_thres.shape[0] * 100
    return acc_thres

def error_metrics(version_id=None,inference_type=None, start_id = 0, end_id = 0,threshold=0.2):
    print(("\t{:>8}" * 4).format("Avg. Error", "Acc", "CD", "F"))#              
    print("test:")            
    count_num = 0
    for j in range(start_id, end_id):   
        if ((j+1-3)%5==0 ):  
            count_num=count_num+1
            print(j+1, end="    ")             
            if count_num%20==0:
                print("")
        else:
            continue              
    print("")

    metrics_np = np.zeros((110,4))    
    count_train = 0
    cd_sum =0
    fscore_sum =0
    abs_error__sum =0
    acc_thres__sum = 0
    for j in range(start_id, end_id):   
        if ((j+1-3)%5==0 ):  
            count_train = count_train + 1
        else:
            continue          

        currentPath = os.getcwd().replace('\\','/')    # 
        # print(currentPath)

        # Load truth ====================================================================================
        gt_pcd = o3d.io.read_point_cloud(currentPath+"/source/"+ str(j+1) + "_source.pcd")                    
        gt_pcd_np = np.asarray(gt_pcd.points)
        pose_select_pcd = o3d.io.read_point_cloud(currentPath+"/source/"+ str(j+1) + "_pose.pcd")                       
        positions_test = (np.asarray(pose_select_pcd.points)).squeeze()
        num = gt_pcd_np.shape[0]
        origin = np.tile(positions_test, (num, 1))
        gt_vec = gt_pcd_np - origin # 
        gt_dist_vec = np.linalg.norm(gt_vec, axis=1) #     

        # Load inference ====================================================================================
        if (inference_type =="one-step"):
            pred_pcd = o3d.io.read_point_cloud(currentPath+"/infer/"+version_id+"_"+str(j+1)+"_one_step.pcd")                    
            print(currentPath+"/infer/"+version_id+"_"+str(j+1)+"_one_step.pcd")                
        else:         
            pred_pcd = o3d.io.read_point_cloud(currentPath+"/infer/"+version_id+"_"+str(j+1)+"_two_step.pcd")    
            print(currentPath+"/infer/"+version_id+"_"+str(j+1)+"_two_step.pcd")                                  
        pred_pcd_np = np.asarray(pred_pcd.points)    

        pred_pcd_np_num = pred_pcd_np.shape[0]
        
        if pred_pcd_np_num<num: 
            print("Reasoning that the point cloud is too few and aligning to the original point cloud.")
            print("gt_pcd_np.shape:",gt_pcd_np.shape)            
            print("pred_pcd_np.shape:",pred_pcd_np.shape)                     
            gt_pcd_np = gt_pcd_np[:pred_pcd_np_num]
            gt_dist_vec = gt_dist_vec[:pred_pcd_np_num]         
            origin = origin[:pred_pcd_np_num]                        
        if pred_pcd_np_num>num: 
            print("Reasoning that the point cloud is too much and aligning to the original point cloud.")
            print("gt_pcd_np.shape:",gt_pcd_np.shape)            
            print("pred_pcd_np.shape:",pred_pcd_np.shape)                           
            pred_pcd_np = pred_pcd_np[:num]

        cd, fscore = eval_pts(pred_pcd_np, gt_pcd_np, threshold=threshold)
        cd_sum = cd_sum + cd 
        fscore_sum = fscore_sum + fscore         
        pred_vec = pred_pcd_np - origin # 
        pred_dist_vec = np.linalg.norm(pred_vec, axis=1) # 
        abs_error_ = abs_error(pred_dist_vec, gt_dist_vec)
        acc_thres_ = acc_thres(pred_dist_vec, gt_dist_vec,threshold=threshold)
        abs_error__sum = abs_error__sum + abs_error_ 
        acc_thres__sum = acc_thres__sum + acc_thres_                 
        print(("\t{: 8.6f}" * 4).format(abs_error_, acc_thres_, cd, fscore))    
        metrics_np[count_train-1, 0 ]=abs_error_
        metrics_np[count_train-1, 1 ]=acc_thres_
        metrics_np[count_train-1, 2 ]=cd
        metrics_np[count_train-1, 3 ]=fscore                   
    
    print(("\t{: 8.6f}" * 4).format(abs_error__sum/count_train, acc_thres__sum/count_train, cd_sum/count_train, fscore_sum/count_train))    
    np.save(currentPath+"/"+version_id+"_metric_tmp",metrics_np) 


if __name__ == '__main__':

    if 1:
        start_id = 1150
        end_id  =  1200
        # inference_type = "one-step"
        inference_type = "two-step"        
        version_id = "version_0"         # originalnerf 
        # version_id = "version_1"             # pc-nerf 
        print("start_id:",start_id,"end_id:",end_id,"inference_type:",inference_type)        

        error_metrics(version_id=version_id, inference_type=inference_type, start_id=start_id, end_id=end_id,threshold=0.2)

