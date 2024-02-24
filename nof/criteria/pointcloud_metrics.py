import open3d as o3d
import numpy as np


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

# def eval_pts(pts1, pts2, threshold=2):
# def eval_pts(pts1, pts2, threshold=1):
# 实验配置 42 
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
