import torch
import numpy as np
from .networks import NOF, Embedding,NOF_coarse,NOF_fine,NOF_plusfine
from nof.criteria import nof_loss #  
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
__all__ = ['render_rays']
import torch
import math

def inference_val(model: NOF, embedding_xy: Embedding, samples_xy: torch.Tensor,rays: torch.Tensor,
            z_vals: torch.Tensor, near_far_child: torch.Tensor, near_far_point: torch.Tensor, range_readings: torch.Tensor,ray_class: torch.Tensor, 
            chunk=1024 * 32, noise_std=1, epsilon=1e-10,isval=False,sub_nerf_test_num=4):
    N_rays = samples_xy.shape[0]
    N_samples = samples_xy.shape[1]
    samples_xy = samples_xy.view(-1, 3)  # shape: (N_rays * N_samples, 3)    
    B = samples_xy.shape[0]  # N_rays * N_samples
    out_chunks = []
    for i in range(0, B, chunk):
        embed_xy = embedding_xy(samples_xy[i:i + chunk])
        out_chunks += [model(embed_xy)]
    out = torch.cat(out_chunks, dim=0)
    prob_occ = out.view(N_rays, N_samples)  # shape: (N_rays, N_samples)   
    prob_free = 1 - prob_occ  # (1-p)  #(N_rays, N_samples)
    prob_free_shift = torch.cat([torch.ones_like(prob_free[:, :1]), prob_free], dim=-1)
    prob_free_cum = torch.cumprod(prob_free_shift, dim=-1)[:, :-1] # shape: (N_rays, N_samples)
    weights_origin = prob_free_cum * prob_occ 
    noise = torch.randn(weights_origin.shape, device=weights_origin.device) * noise_std
    weights_origin = weights_origin + noise
    weights_origin = weights_origin / (torch.sum(weights_origin, dim=-1).reshape(-1, 1) + epsilon)#
 
    depth_final = torch.sum(weights_origin * z_vals, dim=-1)  # shape: (N_rays,) 

    return depth_final, weights_origin

def inference_train(model: NOF, embedding_xy: Embedding, samples_xy: torch.Tensor,rays: torch.Tensor,
            z_vals: torch.Tensor, near_far_child: torch.Tensor, near_far_point: torch.Tensor, range_readings: torch.Tensor,ray_class: torch.Tensor, 
            chunk=1024 * 32, noise_std=1, epsilon=1e-10,isval=False,sub_nerf_test_num=4,use_child_nerf_divide=1,use_child_nerf_loss=0):
    
    N_rays = samples_xy.shape[0]
    N_samples = samples_xy.shape[1]
    samples_xy = samples_xy.view(-1, 3)  # shape: (N_rays * N_samples, 3)    
    B = samples_xy.shape[0]  # N_rays * N_samples
    out_chunks = []
    for i in range(0, B, chunk):
        embed_xy = embedding_xy(samples_xy[i:i + chunk])
        out_chunks += [model(embed_xy)]
    out = torch.cat(out_chunks, dim=0)
    prob_occ = out.view(N_rays, N_samples)  # shape: (N_rays, N_samples)   
    prob_free = 1 - prob_occ  # (1-p)  #(N_rays, N_samples)
    prob_free_shift = torch.cat([torch.ones_like(prob_free[:, :1]), prob_free], dim=-1)
    prob_free_cum = torch.cumprod(prob_free_shift, dim=-1)[:, :-1] # shape: (N_rays, N_samples)
    weights_origin = prob_free_cum * prob_occ 
    # add noise
    noise = torch.randn(weights_origin.shape, device=weights_origin.device) * noise_std
    weights_origin = weights_origin + noise
    # normalize
    weights_origin = weights_origin / (torch.sum(weights_origin, dim=-1).reshape(-1, 1) + epsilon)#
    weights = weights_origin.clone() #  

    if use_child_nerf_loss==1:
        # print("use_child_nerf_loss:",use_child_nerf_loss)
        use_child_free_loss=1
        use_child_depth_loss=1
        use_child_weight = 1
    else:
        # print("use_child_nerf_loss:",use_child_nerf_loss)        
        use_child_free_loss=0
        use_child_depth_loss=0
        use_child_weight = 0


    if use_child_weight:
        mask_in_child_nerf = torch.zeros_like(weights, dtype=torch.bool)
        for i, row in enumerate(z_vals):
            expand_threshold = 0.0            
            interval = near_far_child[i]
            mask_in_child_nerf[i] = (interval[0]-expand_threshold <= row) & (row <= interval[1]+expand_threshold)      
            # while (torch.sum(mask_in_child_nerf[i])==0):
            while (abs(torch.sum(mask_in_child_nerf[i]))==0): #                 
                expand_threshold = expand_threshold + 0.01
                mask_in_child_nerf[i] = (interval[0]-expand_threshold <= row) & (row <= interval[1]+expand_threshold)      
        mask_not_in_child_nerf = ~mask_in_child_nerf
        weights_non_child = weights * mask_not_in_child_nerf.float()  # 
        weights_child = weights * mask_in_child_nerf.float()  # 
        z_vals_child= z_vals * mask_in_child_nerf.float()  #     

        if 1: #  
            for i, row in enumerate(z_vals):
                expand_threshold = 2         #   
                interval = near_far_child[i]
                mask_in_child_nerf[i] = (interval[0]-expand_threshold <= row) & (row <= interval[1]+expand_threshold)      
                while (abs(torch.sum(mask_in_child_nerf[i]))==0): #                 
                    expand_threshold = expand_threshold + 0.01
                    mask_in_child_nerf[i] = (interval[0]-expand_threshold <= row) & (row <= interval[1]+expand_threshold)      
            weights_child = weights * mask_in_child_nerf.float()  # 
            z_vals_child= z_vals * mask_in_child_nerf.float()  #             
        
        
    if use_child_free_loss: 
        weights_free=weights_non_child.reshape(-1,N_samples)#          
        free_num=weights_free.shape[0]        
        if(free_num>=1):    
            if use_child_nerf_divide==1:
                sub_nerf= rays[:, 9].view(-1, 1)
                sub_nerf_tmp=torch.zeros((sub_nerf.shape[0],1),dtype = torch.float,device=sub_nerf.device)     
                child_free_loss=torch.tensor([0],device=sub_nerf.device)                   
                        
                for i in range(sub_nerf_test_num):
                    sub_nerf_tmp[sub_nerf_tmp > 0]=0#         
                    sub_nerf_tmp[torch.logical_and(sub_nerf > (i+0.5), sub_nerf < (i+1.5))]=1 # 
                    count_sub_nerf=torch.sum(sub_nerf_tmp)# 
                    sub_nerf_mask = sub_nerf_tmp.bool()
                    sub_nerf_mask=sub_nerf_mask.reshape(-1,1).repeat(1, N_samples)#     
                    weights_free_sub = weights_free[sub_nerf_mask].reshape(-1, N_samples)#
                    if(count_sub_nerf>=1):
                        child_free_loss=child_free_loss+torch.sum(torch.square(weights_free_sub))/count_sub_nerf   
            else:
                child_free_loss= torch.sum(torch.square(weights_free))/free_num
        else:
            child_free_loss=torch.tensor(0.0)     
    else:
        child_free_loss=torch.tensor(0.0)             

    if use_child_depth_loss:
        smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        weights_child = weights_child / (torch.sum(weights_child, dim=-1).reshape(-1, 1) + epsilon) 
        weights_near=weights_child.reshape(-1,N_samples) 
        near_num=weights_near.shape[0]                
        z_val_child_tmp=z_vals_child.reshape(-1,N_samples) 
        range_readings_child = range_readings.reshape(-1,1)  
        if(near_num>=1):    
            if use_child_nerf_divide==1:                
                sub_nerf= rays[:, 9].view(-1, 1)
                sub_nerf_tmp=torch.zeros((sub_nerf.shape[0],1),dtype = torch.float,device=sub_nerf.device)     
                child_depth_loss=torch.tensor([0],device=sub_nerf.device)                   
                        
                for i in range(sub_nerf_test_num):
                    sub_nerf_tmp[sub_nerf_tmp > 0]=0            
                    sub_nerf_tmp[torch.logical_and(sub_nerf > (i+0.5), sub_nerf < (i+1.5))]=1 
                    count_sub_nerf=torch.sum(sub_nerf_tmp)#    
                    sub_nerf_mask = sub_nerf_tmp.bool()
                    range_readings_child_sub = range_readings_child[sub_nerf_mask].reshape(-1,1)                                
                    sub_nerf_mask = sub_nerf_mask.reshape(-1,1).repeat(1, N_samples)#      
                    weights_near_sub = weights_near[sub_nerf_mask].reshape(-1,N_samples)#

                    z_val_child_tmp_sub = z_val_child_tmp[sub_nerf_mask].reshape(-1,N_samples)       
                    if(count_sub_nerf>=1): 
                        depth_final_sub = torch.sum(weights_near_sub * z_val_child_tmp_sub, dim=-1).reshape(-1,1)
                        child_depth_loss=child_depth_loss+1/count_sub_nerf*0.1* smooth_l1_loss(1e1*depth_final_sub.squeeze(), 1e1*range_readings_child_sub.squeeze())                                 
            else:                
                depth_final = torch.sum(weights_near * z_val_child_tmp, dim=-1)                
                child_depth_loss=1/near_num*0.1* smooth_l1_loss(1e1*depth_final.squeeze(), 1e1*range_readings_child.squeeze())                                   
        else:
            child_depth_loss=torch.tensor(0.0)     
    else:
        child_depth_loss=torch.tensor(0.0)       

    depth_final = torch.sum(weights * z_vals, dim=-1)  # shape: (N_rays,) 
   
    return child_free_loss, child_depth_loss, depth_final, weights_origin


def inference(model: NOF, embedding_xy: Embedding, samples_xy: torch.Tensor,
              z_vals: torch.Tensor, chunk=1024 * 32, noise_std=1, epsilon=1e-10,isval=False):
    """
    Helper function that performs model inference.

    :param model: NOF mode
    :param embedding_xy: position embedding module
    :param samples_xy: sampled position (shape: (N_rays, N_samples, 2))
    :param z_vals: depths of the sampled positions (shape: N_rays, N_samples)
    :param chunk: the chunk size in batched inference (default: 1024*32)
    :param noise_std: factor to perturb the model's prediction of sigma (default: 1)
    :param epsilon: a small number to avoid the 0 of weights_sum (default: 1e-10)

    :return:
        depth_final: rendered range value for each Lidar beams (shape: (N_rays,))
        weights: weights of each sample (shape: (N_rays, N_samples))
        opacity: the cross entropy of the predicted occupancy values 
    """
    N_rays = samples_xy.shape[0]
    N_samples = samples_xy.shape[1]

    # Embed directions
    # samples_xy = samples_xy.view(-1, 2)  # shape: (N_rays * N_samples, 2)
    samples_xy = samples_xy.view(-1, 3)  # shape: (N_rays * N_samples, 3)    

    # prediction, to get rangepred and raw sigma
    B = samples_xy.shape[0]  # N_rays * N_samples
    # print("samples_xy.shape:\n",samples_xy.shape)#add 
    out_chunks = []
    for i in range(0, B, chunk):
        # embed position by chunk
        embed_xy = embedding_xy(samples_xy[i:i + chunk])
        # embed_xy = samples_xy[i:i + chunk]
        out_chunks += [model(embed_xy)]

    out = torch.cat(out_chunks, dim=0)
    prob_occ = out.view(N_rays, N_samples)  # shape: (N_rays, N_samples)   

    # Volume Rendering: synthesis the 2d lidar scan
    prob_free = 1 - prob_occ  # (1-p)  #(N_rays, N_samples)
    
    # ((N_rays, 1) + (N_rays, N_samples)=(N_rays, N_samples+1)
    prob_free_shift = torch.cat(
        [torch.ones_like(prob_free[:, :1]), prob_free], dim=-1
    )
    prob_free_cum = torch.cumprod(prob_free_shift, dim=-1)[:, :-1] # shape: (N_rays, N_samples)
    weights = prob_free_cum * prob_occ 

    # add noise
    noise = torch.randn(weights.shape, device=weights.device) * noise_std
    weights = weights + noise

    # normalize
    if((isval==False)):
        weights = weights / (torch.sum(weights, dim=-1).reshape(-1, 1) + epsilon)#
    depth_final = torch.sum(weights * z_vals, dim=-1)  # shape: (N_rays,) 

    # opacity regularization
    opacity = torch.mean(torch.log(0.1 + prob_occ) + torch.log(0.1 + prob_free) + 2.20727)

    return depth_final, weights, opacity


def inference_0525_2(model: NOF, embedding_xy: Embedding, samples_xy: torch.Tensor,
              z_vals: torch.Tensor, other_interest_sub_nerf_number: torch.Tensor, near_far_child: torch.Tensor,
              chunk=1024 * 32, noise_std=1, epsilon=0,isval=False,is_fine=0, depth_inference_method=0):    
    N_rays = samples_xy.shape[0]
    N_samples = samples_xy.shape[1]
    samples_xy = samples_xy.view(-1, 3)  # shape: (N_rays * N_samples, 3)    
    B = samples_xy.shape[0]  # N_rays * N_samples
    out_chunks = []
    for i in range(0, B, chunk):
        embed_xy = embedding_xy(samples_xy[i:i + chunk])
        out_chunks += [model(embed_xy)]
    out = torch.cat(out_chunks, dim=0)
    prob_occ = out.view(N_rays, N_samples)  # shape: (N_rays, N_samples)   
    prob_free = 1 - prob_occ  # (1-p)  #(N_rays, N_samples)
    prob_free_shift = torch.cat([torch.ones_like(prob_free[:, :1]), prob_free], dim=-1)
    prob_free_cum = torch.cumprod(prob_free_shift, dim=-1)[:, :-1] # shape: (N_rays, N_samples)
    weights = prob_free_cum * prob_occ #   # shape: (N_rays, N_samples)
    weights = weights / (torch.sum(weights, dim=-1).reshape(-1, 1) + epsilon)#    

    mask_child = torch.zeros_like(weights, dtype=torch.bool)
    child_bound = torch.zeros((N_rays,2),dtype = torch.float,device=weights.device)    

    if 1: #  
        for i, row in enumerate(z_vals):
            expand_threshold = 0.01            
            interval = near_far_child[i]
            mask_child[i] = (interval[0] - expand_threshold < row ) & (row < interval[1] + expand_threshold)
            child_bound[i][0] =  interval[0] - expand_threshold           
            child_bound[i][1] =  interval[1] + expand_threshold                  
            flag = 0
            while (torch.sum(mask_child[i])==0):
                expand_threshold = expand_threshold + 0.01
                mask_child[i] = (interval[0] - expand_threshold < row) & (row < interval[1] + expand_threshold)    
                child_bound[i][0] =  interval[0] - expand_threshold           
                child_bound[i][1] =  interval[1] + expand_threshold                        

    mask0 = torch.zeros_like(weights, dtype=torch.bool)
    for i, row in enumerate(z_vals):
        interval = near_far_child[i]
        mask0[i] = (1 <= row) & (row < interval[1]+0.1 )        

    print("depth_inference_method:",depth_inference_method)
    if(depth_inference_method==2):
        use_parent_bound = 0
        use_peak_child_bound_1 = 1
    else:
        use_parent_bound = 1
        use_peak_child_bound_1 = 0
        
    use_peak_search7 = 1
    use_accumulate_weights = 0
    
    rays_effective_flag=torch.zeros((N_rays,1),dtype = torch.bool,device=weights.device)       # 
    if(use_accumulate_weights): # 
        weights_for_sum0 = weights * mask0.float()  
        weights_sum0 = torch.sum(weights_for_sum0, dim=-1).reshape(-1, 1)                
        i=0
        while i<N_rays:            
            if(abs(other_interest_sub_nerf_number[i]-0)<0.5): # 
                rays_effective_flag[i]=1   
                i=i+1
            elif(other_interest_sub_nerf_number[i]>0.5): 
                max_weight_sum_index=i
                if(weights_sum0[i]<0.50 ):                         
                    for j in range(0, int(other_interest_sub_nerf_number[i])):
                        if(weights_sum0[max_weight_sum_index]<weights_sum0[i+j+1]):
                            max_weight_sum_index=i+j+1                                
                rays_effective_flag[max_weight_sum_index]=1      
                
                i=i+int(other_interest_sub_nerf_number[i]) 
            else:
                i=i+1

    if(use_peak_search7): 
        weights_smothed = torch.zeros_like(weights, dtype=torch.float)
        i=0
        while i<N_rays:         
            weights_smothed[i] = torch.tensor(gaussian_filter(weights[i].cpu().numpy(), sigma=5))    
            i = i+1
        max_indices = torch.argmax(weights_smothed, dim=1)# 

        mask1 = torch.zeros_like(weights, dtype=torch.bool)
        mask1[torch.arange(weights.shape[0]), max_indices] = True
        mask2=(mask_child.float()) * (mask1.float()) #  
        mask2 = torch.sum(mask2, dim=-1).reshape(-1, 1) 
        weights_child_here = weights * mask_child.float()  #
        weights_child_here_sum = torch.sum(weights_child_here, dim=-1).reshape(-1, 1) 

        i=0
        while i<N_rays:            
            if(abs(other_interest_sub_nerf_number[i]-0)<0.5): # 
                rays_effective_flag[i]=True       
                i=i+1
            elif(other_interest_sub_nerf_number[i]>0.5): 
                if (abs(mask2[i]-1)<0.1):
                    max_weight_sum_index=i
                else:
                    max_weight_sum_index=i
                    exist_peak = 0
                    for j in range(0, int(other_interest_sub_nerf_number[i])):
                        if (abs(mask2[i+j+1]-1)<0.1): 
                            max_weight_sum_index=i+j+1                                 
                            exist_peak = 1
                            break
                    if exist_peak==0: 
                        for j in range(0, int(other_interest_sub_nerf_number[i])):
                            if( weights_child_here_sum[i+j+1] >weights_child_here_sum[max_weight_sum_index]):
                                max_weight_sum_index=i+j+1     
                rays_effective_flag[max_weight_sum_index] = True                   
                i=i+int(other_interest_sub_nerf_number[i])+1              
            else:
                i=i+1                
                
    if use_parent_bound:
        depth_final = torch.sum(weights * z_vals, dim=-1)  
        
    if use_peak_child_bound_1:  
        weights_child_new = weights * mask_child.float()  
        weights_child_new = weights_child_new / (torch.sum(weights_child_new, dim=-1).reshape(-1, 1) + epsilon) 
        depth_final = torch.sum(weights_child_new * z_vals, dim=-1)  
        for i in range(0, depth_final.shape[0]):
            if abs(depth_final[i]-0)<0.1:
                print("error")
                print("depth_final[i]:",depth_final[i])

    opacity = torch.mean(torch.log(0.1 + prob_occ) + torch.log(0.1 + prob_free) + 2.20727)

    if 0:
        i = 0
        while i<N_rays:                    
            if rays_effective_flag[i] ==True:
                if depth_final[i]<child_bound[i][0]:                    
                    print("The composite depth value obtained by using the peak value of the child nerf aabb box is not located within the subfield where it is located")
                    print("error")
                if depth_final[i]>child_bound[i][1]:                    
                    print("The composite depth value obtained by using the peak value of the child nerf aabb box is not located within the subfield where it is located")
                    print("error")                    
            i = i + 1

    return depth_final, weights, opacity,rays_effective_flag


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)#
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples， 
    if det: #False
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    u = u.contiguous()  
    u=u.to("cuda:0")    
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)  

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples



def render_rays_train(model: NOF,model_fine: NOF, embedding_xy: Embedding, rays: torch.Tensor,sub_nerf_test_num=4,
                N_samples=64,N_importance=128, use_disp=False, perturb=0, noise_std=1, chunk=1024 * 3, isval=False, issegmentated=0,childnerf_ratio=0.5,use_child_nerf_divide=0,
                use_child_nerf_loss=0):            
    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]  # shape: (N_rays, 2)
    near, far = rays[:, 6].view(-1, 1), rays[:, 7].view(-1, 1)  # shape: (N_rays, 1)  
    ray_class= rays[:, 8].view(-1, 1) ##
    near_far_child=rays[:, 10:12].view(-1, 2) # 
    near_child, far_child = rays[:, 10].view(-1, 1), rays[:, 11].view(-1, 1) # 
    near_far_point=rays[:, 12:14].view(-1, 2) # 
    range_readings= rays[:, -1].view(-1, 1)# 
    
    if issegmentated == 0: 
        z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # shape: (N_samples,)
        z_steps = z_steps.expand(N_rays, N_samples)
        z_vals = near * (1 - z_steps) + far * z_steps
    else:
        parent_bound_num = int(N_samples*(1-childnerf_ratio))
        child_bound_num = N_samples - parent_bound_num
        z_steps_parent = torch.linspace(0, 1, parent_bound_num, device=rays.device)  
        z_steps_parent = z_steps_parent.expand(N_rays, parent_bound_num)
        z_vals_parent  = near * (1 - z_steps_parent) + far * z_steps_parent
        z_steps_child = torch.linspace(0, 1, child_bound_num, device=rays.device)  
        z_steps_child = z_steps_child.expand(N_rays, child_bound_num)
        z_vals_child  = near_child * (1 - z_steps_child) + far_child * z_steps_child
        z_vals, _ = torch.sort(torch.cat([z_vals_parent, z_vals_child], -1), -1)
    
    if 0:# 
        z_steps=torch.sqrt(z_steps)  
        z_vals = near * (1 - z_steps) + far * z_steps
    
    # perturb sampling depths (z_vals)
    if perturb > 0:#
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # shape: (N_rays, N_samples-1)
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], dim=-1)
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    epsilon = 1e-10
    # epsilon = 0    
    samples_xy = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # shape: (N_rays, N_samples, 3) #rt=o+td     
    child_free_loss, child_depth_loss, depth_final, weights_origin= \
        inference_train(model, embedding_xy, samples_xy, rays, z_vals,near_far_child,near_far_point, range_readings, \
            ray_class, chunk, noise_std, epsilon,isval,sub_nerf_test_num=sub_nerf_test_num,use_child_nerf_divide= use_child_nerf_divide,use_child_nerf_loss=use_child_nerf_loss)               

    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    # z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=False)
    z_samples = sample_pdf(z_vals_mid, weights_origin[...,1:-1], N_importance, det=(perturb==0.), pytest=False)    
    z_samples = z_samples.detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    samples_xy = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # shape: (N_rays, N_samples, 3) #rt=o+td       
    child_free_loss_fine, child_depth_loss_fine, depth_final_fine,weights_origin_fine = \
    inference_train(model_fine, embedding_xy, samples_xy, rays, z_vals,near_far_child,near_far_point, range_readings, \
            ray_class, chunk, noise_std, epsilon, isval,sub_nerf_test_num=sub_nerf_test_num,use_child_nerf_divide=use_child_nerf_divide,use_child_nerf_loss=use_child_nerf_loss)   

    results = {
                'child_free_loss_fine': child_free_loss_fine,
                'child_depth_loss_fine': child_depth_loss_fine,                
               "depth_fine":depth_final_fine,
                'child_free_loss': child_free_loss,
                'child_depth_loss': child_depth_loss,                        
               'depth': depth_final,
               }

    return results

    
def render_rays_val(model: NOF,model_fine: NOF, embedding_xy: Embedding, rays: torch.Tensor,sub_nerf_test_num=4,
                N_samples=64,N_importance=128, use_disp=False, perturb=0, noise_std=1, chunk=1024 * 3, isval=False):               

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]  # shape: (N_rays, 2)
    near, far = rays[:, 6].view(-1, 1), rays[:, 7].view(-1, 1)  # shape: (N_rays, 1)  
    ray_class= rays[:, 8].view(-1, 1) ##   
    near_far_child=rays[:, 10:12].view(-1, 2) #
    near_far_point=rays[:, 12:14].view(-1, 2) # 
    range_readings= rays[:, -1].view(-1, 1)# 
    
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # shape: (N_samples,)
    z_steps = z_steps.expand(N_rays, N_samples)
    z_vals = near * (1 - z_steps) + far * z_steps
    
    if 0:#
        z_steps=torch.sqrt(z_steps)  
        z_vals = near * (1 - z_steps) + far * z_steps

    # perturb sampling depths (z_vals)
    if perturb > 0:#
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # shape: (N_rays, N_samples-1)
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], dim=-1)
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    # epsilon = 0
    epsilon = 1e-10
    samples_xy = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # shape: (N_rays, N_samples, 3) #rt=o+td     
    depth_final, weights_origin= \
        inference_val(model, embedding_xy, samples_xy, rays, z_vals,near_far_child,near_far_point, range_readings, \
            ray_class, chunk, noise_std, epsilon=epsilon, isval=isval,sub_nerf_test_num=sub_nerf_test_num)               

    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    # z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=False)
    z_samples = sample_pdf(z_vals_mid, weights_origin[...,1:-1], N_importance, det=(perturb==0.), pytest=False)    
    z_samples = z_samples.detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    samples_xy = rays_o.unsqueeze(1) + \
                 rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # shape: (N_rays, N_samples, 3) #rt=o+td       
    depth_final_fine, weights_origin_fine = \
    inference_val(model_fine, embedding_xy, samples_xy, rays, z_vals,near_far_child,near_far_point, range_readings, \
            ray_class, chunk, noise_std,  epsilon=epsilon, isval=isval,sub_nerf_test_num=sub_nerf_test_num)   

    results = {
               "depth_fine":depth_final_fine,
               'depth': depth_final,
               }

    return results

def render_rays(model: NOF,model_fine: NOF, embedding_xy: Embedding, rays: torch.Tensor,
                N_samples=64,N_importance=128, use_disp=False, perturb=0, noise_std=1, chunk=1024 * 3, isval=False):               
    """
    Render rays by computing the output of @model applied on @rays

    :param model: NOF model, defined by models.NOF()
    :param embedding_xy: embedding model for position, defined by models.Embedding()

    :param rays: the input data, include: ray original, directions, near and far depth bounds
                 (shape: (N_rays, 3+3+2))
    :param N_samples: number of samples pre ray (default: 64)
    :param use_disp: whether to sample in disparity space (inverse depth) (default: False)
    :param perturb: factor to perturb the sampling position on the ray 
                    (0 for default, for coarse model only)
    :param noise_std: factor to perturb the model's prediction of sigma (1 for default) 
    :param chunk: the chunk size in batched inference

    :return result: dictionary containing final range value, weights and opacity
    """
    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]  # shape: (N_rays, 2)
    near, far = rays[:, 6].view(-1, 1), rays[:, 7].view(-1, 1)  # shape: (N_rays, 1)

    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # shape: (N_samples,)
    z_steps = z_steps.expand(N_rays, N_samples)

    if use_disp:#
        # linear sampling in disparity space
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)
    else:
        # linear sampling in depth space
        z_vals = near * (1 - z_steps) + far * z_steps

    # perturb sampling depths (z_vals)
    if perturb > 0:#
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # shape: (N_rays, N_samples-1)
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], dim=-1)

        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    samples_xy = rays_o.unsqueeze(1) + \
                 rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # shape: (N_rays, N_samples, 3) #rt=o+td  
    
    depth, weights, opacity = \
        inference(model, embedding_xy, samples_xy, z_vals, chunk, noise_std, isval)

    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=False)
    z_samples = z_samples.detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    # pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
    samples_xy = rays_o.unsqueeze(1) + \
                 rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # shape: (N_rays, N_samples, 3) #rt=o+td      

    depth_fine, weights, opacity_fine = \
        inference(model_fine, embedding_xy, samples_xy, z_vals, chunk, noise_std, isval)#

    weights_mask=weights.argsort(dim=-1, descending=True).eq(weights.shape[1]-1)#
    weights_mask=weights_mask.bool()
    depth2=z_vals[weights_mask]#

    results = {'depth_fine': depth_fine,
               'weights': weights,
               'opacity': opacity,
               'z_vals':z_vals,
               "depth":depth,#
               "depth2":depth2,#
               "opacity_fine":opacity_fine 
               }

    return results


def render_rays_view_0525_2_2(model: NOF,model_fine: NOF, 
                          embedding_xy: Embedding, rays: torch.Tensor,other_interest_sub_nerf_number: torch.Tensor,
                N_samples=64,N_importance=128, use_disp=False, perturb=0, noise_std=1, chunk=1024 * 3, isval=False, depth_inference_method=0):               
    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]  # shape: (N_rays, 2)
    # near, far = rays[:, 6].view(-1, 1), rays[:, 7].view(-1, 1)  # shape: (N_rays, 1)
    
    near, far = rays[:, 9].view(-1, 1), rays[:, 10].view(-1, 1)  #
    near_far_child = rays[:, 6:8].view(-1, 2)  

    if 1: 
        z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # shape: (N_samples,)
        z_steps = z_steps.expand(N_rays, N_samples)
        z_vals = near * (1 - z_steps) + far * z_steps
    else: #     
        near_child, far_child = rays[:, 6].view(-1, 1), rays[:, 7].view(-1, 1) #
        print("near[0]:",near[0],"near_child[0]:",near_child[0],"far_child[0]:",far_child[0],"far[0]:",far[0])        
        parent_bound_num = int(N_samples*0.5)
        child_bound_num = N_samples - parent_bound_num

        z_steps_parent = torch.linspace(0, 1, parent_bound_num, device=rays.device)  
        z_steps_parent = z_steps_parent.expand(N_rays, parent_bound_num)
        z_vals_parent  = near * (1 - z_steps_parent) + far * z_steps_parent

        z_steps_child = torch.linspace(0, 1, child_bound_num, device=rays.device)  
        z_steps_child = z_steps_child.expand(N_rays, child_bound_num)
        z_vals_child  = near_child * (1 - z_steps_child) + far_child * z_steps_child

        z_vals, _ = torch.sort(torch.cat([z_vals_parent, z_vals_child], -1), -1)        

    if perturb > 0:#
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # shape: (N_rays, N_samples-1)
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], dim=-1)

        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    samples_xy = rays_o.unsqueeze(1) + \
                 rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # shape: (N_rays, N_samples, 3) #rt=o+td  

    epsilon = 1e-10  # 1e-10 
    is_fine = 0
    depth, weights, opacity,rays_effective_flag = \
        inference_0525_2(model, embedding_xy, samples_xy, z_vals,
                       other_interest_sub_nerf_number, near_far_child,  chunk, noise_std, epsilon, isval,is_fine,depth_inference_method)

    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=False)
    z_samples = z_samples.detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    samples_xy = rays_o.unsqueeze(1) + \
                 rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # shape: (N_rays, N_samples, 3) #rt=o+td      

    is_fine = 1
    depth_fine, weights, opacity_fine,rays_effective_flag_fine = \
        inference_0525_2(model_fine, embedding_xy, samples_xy, z_vals, 
                    other_interest_sub_nerf_number, near_far_child, chunk, noise_std, epsilon, isval,is_fine,depth_inference_method)    
    
    points_inference= torch.ones((rays_o.shape[0],3),dtype=torch.float,device=rays_o.device)     
    for k in range(rays_o.shape[0]):
        points_inference[k][0]=rays_o[k][0]+depth[k]*rays_d[k][0]
        points_inference[k][1]=rays_o[k][1]+depth[k]*rays_d[k][1]
        points_inference[k][2]=rays_o[k][2]+depth[k]*rays_d[k][2]         
    
    points_inference_fine= torch.ones((rays_o.shape[0],3),dtype=torch.float,device=rays_o.device)  
    for k in range(rays_o.shape[0]):
        points_inference_fine[k][0]=rays_o[k][0]+depth_fine[k]*rays_d[k][0]
        points_inference_fine[k][1]=rays_o[k][1]+depth_fine[k]*rays_d[k][1]
        points_inference_fine[k][2]=rays_o[k][2]+depth_fine[k]*rays_d[k][2]     

    results = {'depth_fine': depth_fine,
               'weights': weights,
               'opacity': opacity,
               'z_vals':z_vals,
               "depth":depth,#
               "opacity_fine":opacity_fine,
               "points_inference_fine":points_inference_fine, 
               "points_inference":points_inference,         

               "rays_effective_flag":rays_effective_flag, 
               "rays_effective_flag_fine":rays_effective_flag_fine                               
               }

    return results

