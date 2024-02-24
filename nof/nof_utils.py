"""The functions for some tools
"""

import torch
from torch.optim import SGD, Adam
import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_path', type=str, default=None,
                        help='')           
    parser.add_argument('--re_loaddata', type=int, default=0,
                        help='Whether the ray data needs to be regenerated')           
    parser.add_argument('--datasettype', type=str, default='kitti_sequence00_repeat',
                        help='')    
    parser.add_argument('--root_dir', type=str, default='/media/bit/T7/dataset/kitti/dataset/sequences/00/pcd',
                        help='root directory of dataset')    
    parser.add_argument('--pose_path', type=str, default='/media/bit/T7/dataset/kitti/dataset/sequences/00/poses.txt',
                        help='')       
    parser.add_argument('--data_start', type=int, default=1,
                        help='Starting serial number of the pcd file')
    parser.add_argument('--data_end', type=int, default=2,
                        help='Ending serial number of the pcd file')          
    parser.add_argument('--parentnerf_path', type=str, 
                        default='/home/meng/subject/ir-mcl-3DLidar03-19/logs/nof_maicity/maicity_multi_frame5/数据配置3/radius_outlier1/inliers_4.pcd',
                        help='Path to store the point cloud in the parent nerf aabb box')         
    parser.add_argument('--subnerf_path', type=str, 
                        default='/home/meng/subject/ir-mcl-3DLidar03-19/logs/nof_maicity/maicity_multi_frame5/数据配置0/sub_pointcloud/sub_nerf2',
                        help='Path to store the point cloud in the child nerf aabb box')         
    parser.add_argument('--sub_nerf_test_num', type=int, default=3,
                        help='Number of child nerf aabb boxes used for testing')  
    parser.add_argument('--range_delete_x', type=float, default=2,
                        help='Range of occupancy of the vehicle body in the x-direction')
    parser.add_argument('--range_delete_y', type=float, default=1,
                        help='Range of occupancy of the vehicle body in the y-direction')    
    parser.add_argument('--range_delete_z', type=float, default=0.5,
                        help='Range of occupancy of the vehicle body in the z-direction')                       
    parser.add_argument('--over_height', type=float, default=0.168,
                        help='over_height')      
    parser.add_argument('--over_low', type=float, default=-2.0,
                        help='over_low')      
    parser.add_argument('--interest_x', type=float, default=12,
                        help='Select point clouds in the vicinity of the vehicle traveling path.')    
    parser.add_argument('--interest_y', type=float, default=10,
                        help='Select point clouds in the vicinity of the vehicle traveling path.')        
    parser.add_argument('--cloud_size_val', type=int, default=128,
                        help='')
    parser.add_argument('--surface_expand', type=float, default=0.5,
                        help='')    
    parser.add_argument('--nerf_length_min', type=float, default=-4.5,
                        help='Minimum value in the direction of the length of the parent nerf aabb box (coordinates)')        
    parser.add_argument('--nerf_length_max', type=float, default= 25.5,
                        help='Maximum value in the direction of the length of the parent nerf aabb box (coordinates)')            
    parser.add_argument('--nerf_width_min', type=float, default=-4.5,
                        help='Minimum value in the direction of the width of the parent nerf aabb box (coordinates)')        
    parser.add_argument('--nerf_width_max', type=float, default= 25.5,
                        help='Maximum value in the direction of the width of the parent nerf aabb box (coordinates)')           
    parser.add_argument('--nerf_height_min', type=float, default=-2.0,
                        help='Minimum value in the direction of the height of the parent nerf aabb box (coordinates)')    
    parser.add_argument('--nerf_height_max', type=float, default=0.5,
                        help='Maximum value in the direction of the height of the parent nerf aabb box (coordinates)')

    parser.add_argument('--L_pos', type=int, default=10,
                        help='the frequency of the positional encoding.')    
    parser.add_argument('--feature_size', type=int, default=256,
                        help='the dimension of the feature maps.')
    parser.add_argument('--use_skip', default=True, action="store_true",
                        help='use skip architecture')    
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
        
    parser.add_argument('--exp_name', type=str, default='nof_kitti/sequence00',
                        help='experiment name')    
    parser.add_argument('--seed', type=int, default=42,
                        help='set a seed for fairly comparison during training')
    parser.add_argument('--loss_type', type=str, default='smoothl1',
                        choices=['mse', 'l1', 'smoothl1'],
                        help='loss to use')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=12,
                        help='')              
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')    
    parser.add_argument('--chunk', type=int, default=32 * 1024,
                        help='chunk size to split the input to avoid OOM')#OOM,out of memory 
    parser.add_argument('--num_epochs', type=int, default=16,
                        help='number of training epochs')
    parser.add_argument('--decay_step', nargs='+', type=int, default=[200],
                        help='scheduler decay step')
    parser.add_argument('--decay_epochs', nargs='+', type=int, default=[2],
                        help='scheduler decay epoch')                        
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')    

    parser.add_argument('--use_child_nerf_divide', type=int, default=0,
                        help='Whether or not to use child nerf spatialization')       
    parser.add_argument('--use_child_nerf_loss', type=int, default=0,
                        help='Whether or not to use child nerf related losses')       
    parser.add_argument('--use_segmentated_sample', type=int, default=0,
                        help='Whether to use segmented sampling on the ray')           
    parser.add_argument('--segmentated_child_nerf_ratio', type=float, default=0.5,
                        help='If segmented sampling is used, the proportion of child nerf segmented sampling')      
    parser.add_argument('--lambda_loss', type=float, default=0.5,
                        help='Coarse network predicts weights for loss of depth')       
    parser.add_argument('--lambda_loss_fine', type=float, default=0.5,
                        help='fine network predicts weights for loss of depth')       
    parser.add_argument('--lambda_child_free_loss', type=float, default=0.5,
                        help='child_free_loss weight')       
    parser.add_argument('--lambda_child_depth_loss', type=float, default=0.5,
                        help='child_depth_loss weight')     

    parser.add_argument('--N_samples', type=int, default=128,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=256,
                        help='number of fine samples')                        #
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='std dev of noise added to regularize sigma')    
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')    

    parser.add_argument('--visualize', type=int, default=1,
                        help='Whether or not to visualize training set loss during training')    
    parser.add_argument('--current_epoch', type=int, default=0,
                        help='Current epoch')    
    parser.add_argument('--saveploty_path', type=str, default='./logs/nof_kitti/sequence00/ploty0002',
                        help='Address of the file where train/loss is saved')              
    parser.add_argument('--saveploty_path_range', type=str, default='./logs/nof_kitti/sequence00/ploty0002',
                        help='Address of the file where train/loss is saved')        
    parser.add_argument('--saveploty_path_range_fine', type=str, default='./logs/nof_kitti/sequence00/ploty0002',
                        help='Address of the file where train/loss is saved')       
    parser.add_argument('--saveploty_path_child_free', type=str, default='./logs/nof_kitti/sequence00/ploty0002',
                        help='Address of the file where train/loss is saved')     
    parser.add_argument('--saveploty_path_child_free_fine', type=str, default='./logs/nof_kitti/sequence00/ploty0002',
                        help='Address of the file where train/loss is saved')     
    parser.add_argument('--saveploty_path_child_depth', type=str, default='./logs/nof_kitti/sequence00/ploty0002',
                        help='Address of the file where train/loss is saved')     
    parser.add_argument('--saveploty_path_child_depth_fine', type=str, default='./logs/nof_kitti/sequence00/ploty0002',
                        help='Address of the file where train/loss is saved')       

    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')

    return parser.parse_args()


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_optimizer(hparams, parameters):
    eps = 1e-8
    if hparams.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=hparams.lr,
                        momentum=hparams.momentum, weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'adam':
        optimizer = Adam(parameters, lr=hparams.lr, eps=eps,
                         weight_decay=hparams.weight_decay)
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer


def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint:  # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name) + 1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    # print("model_name:",model_name)
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


def decode_batch(batch):
    rays = batch['rays']  # shape: (B, 6)
    rangs = batch['ranges']  # shape: (B, 1)
    return rays, rangs

def decode_batch2(batch):
    rays = batch['rays']  # shape: (B, 6)
    # rangs = batch['ranges']  # shape: (B, 1)
    return rays
