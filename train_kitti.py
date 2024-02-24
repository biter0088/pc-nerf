import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from nof.dataset import nof_dataset
from nof.networks import Embedding, NOF_fine,NOF_coarse
from nof.render import render_rays_val, render_rays_train
from nof.criteria import nof_loss
from nof.criteria.metrics import abs_error, acc_thres, eval_points
from nof.nof_utils import get_opts, get_learning_rate, get_optimizer, decode_batch,load_ckpt
# plot
import matplotlib.pyplot as plt
import numpy as np 

class NOFSystem(LightningModule):
    def __init__(self, hparams):
        super(NOFSystem, self).__init__()
        self.save_hyperparameters(hparams)

        self.embedding_position = Embedding(in_channels=3, N_freq=self.hparams.L_pos)#
        self.nof_coarse = NOF_coarse(feature_size=self.hparams.feature_size,
                       in_channels_xy=3 + 3 * self.hparams.L_pos * 2,
                       use_skip=self.hparams.use_skip)         #                        
        self.nof_fine = NOF_fine(feature_size=self.hparams.feature_size,
                       in_channels_xy=3 + 3 * self.hparams.L_pos * 2,
                       use_skip=self.hparams.use_skip)         # 
        if(self.hparams.ckpt_path):
            print("Load pre trained weights")
            nof_ckpt = self.hparams.ckpt_path
            load_ckpt(self.nof_coarse, nof_ckpt, model_name='nof_coarse')
            load_ckpt(self.nof_fine, nof_ckpt, model_name='nof_fine')                

        self.loss = nof_loss[self.hparams.loss_type]()#
        self.loss2 = nof_loss[self.hparams.loss_type]()#        
        self.plotx=[]#        
        self.ploty=[]#       
        self.plot_loss_range=[]#
        self.plot_loss_range_fine=[]#      
        self.plot_loss_child_free=[]#
        self.plot_loss_child_free_fine=[]#        
        self.plot_loss_child_depth=[]#
        self.plot_loss_child_depth_fine=[]#          
        plt.ion()#

    def prepare_data(self):
        dataset = nof_dataset[self.hparams.datasettype]       
        if self.hparams.datasettype=="kitti_dataload":   
            kwargs = {'root_dir': self.hparams.root_dir, 'data_start':self.hparams.data_start, 'data_end':self.hparams.data_end, 'cloud_size_val':self.hparams.cloud_size_val,
                                'range_delete_x':self.hparams.range_delete_x, 'range_delete_y':self.hparams.range_delete_y, 'range_delete_z':self.hparams.range_delete_z,
                                'sub_nerf_test_num':self.hparams.sub_nerf_test_num,
                                'parentnerf_path':self.hparams.parentnerf_path, 
                                'pose_path':self.hparams.pose_path,
                                'subnerf_path':self.hparams.subnerf_path,
                                'surface_expand':self.hparams.surface_expand,
                                'interest_x':self.hparams.interest_x, 'interest_y':self.hparams.interest_y,                  
                                'over_height':self.hparams.over_height, 'over_low':self.hparams.over_low,         
                                're_loaddata':self.hparams.re_loaddata,
                                'result_path':self.hparams.result_path                                 
                                }
        if self.hparams.datasettype=="maicity_dataload":   
            kwargs = {'root_dir': self.hparams.root_dir, 'data_start':self.hparams.data_start, 'data_end':self.hparams.data_end, 'cloud_size_val':self.hparams.cloud_size_val,
                                'range_delete_x':self.hparams.range_delete_x, 'range_delete_y':self.hparams.range_delete_y, 'range_delete_z':self.hparams.range_delete_z,
                                'sub_nerf_test_num':self.hparams.sub_nerf_test_num,
                                'nerf_length_min':self.hparams.nerf_length_min, 'nerf_length_max':self.hparams.nerf_length_max,
                                'nerf_width_min':self.hparams.nerf_width_min, 'nerf_width_max':self.hparams.nerf_width_max,
                                'nerf_height_min':self.hparams.nerf_height_min, 'nerf_height_max':self.hparams.nerf_height_max, 
                                'pose_path':self.hparams.pose_path,
                                'subnerf_path':self.hparams.subnerf_path,
                                'surface_expand':self.hparams.surface_expand,
                                're_loaddata':self.hparams.re_loaddata,                                
                                'result_path':self.hparams.result_path                                                          
                                }     
        self.train_dataset = dataset(split='train',  **kwargs)
        self.val_dataset = dataset(split='val',  **kwargs)#        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, num_workers=16,
                          batch_size=self.hparams.batch_size, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, num_workers=4,
                          batch_size=self.hparams.batch_size_val, pin_memory=True)

    def forward(self, rays, isval):        
        if isval==False:
            rendered_rays = render_rays_train(
                model=self.nof_coarse,model_fine=self.nof_fine, embedding_xy=self.embedding_position, rays=rays,
                N_samples=self.hparams.N_samples, N_importance=self.hparams.N_importance, use_disp=self.hparams.use_disp,
                perturb=self.hparams.perturb, noise_std=self.hparams.noise_std,
                chunk=self.hparams.chunk, isval=isval,sub_nerf_test_num=self.hparams.sub_nerf_test_num,
                issegmentated = self.hparams.use_segmentated_sample,  childnerf_ratio = self.hparams.segmentated_child_nerf_ratio,
                use_child_nerf_divide = self.hparams.use_child_nerf_divide,use_child_nerf_loss = self.hparams.use_child_nerf_loss
            )     

        if isval==True:       
            rendered_rays = render_rays_val(
                model=self.nof_coarse,model_fine=self.nof_fine, embedding_xy=self.embedding_position, rays=rays,
                N_samples=self.hparams.N_samples, N_importance=self.hparams.N_importance, use_disp=self.hparams.use_disp,
                perturb=self.hparams.perturb, noise_std=self.hparams.noise_std,
                chunk=self.hparams.chunk, isval=isval,sub_nerf_test_num=self.hparams.sub_nerf_test_num
            )                        
        return rendered_rays

    def configure_optimizers(self):
        parameters = []
        parameters += list(self.nof_coarse.parameters())
        parameters += list(self.nof_fine.parameters())#
        self.optimizer = get_optimizer(self.hparams, parameters)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[5, 120, 256],
                                     gamma=self.hparams.decay_gamma)             
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        self.log('lr', get_learning_rate(self.optimizer))
        if(batch_idx==1):#
            print("get_learning_rate(self.optimizer):",get_learning_rate(self.optimizer))
        rays, gt_ranges = decode_batch(batch)
        results = self.forward(rays, False)#                

        pred_ranges_fine = results['depth_fine']    # 
        pred_ranges = results['depth']     #         torch.Size([batch_size])  #torch.Size([N_rays])
        
        loss_range=torch.tensor([0],device=rays.device)                   
        loss_range_fine=torch.tensor([0],device=rays.device)                             
        if self.hparams.use_child_nerf_divide==1: # 
            sub_nerf= rays[:, 9].view(-1, 1)
            sub_nerf=sub_nerf.squeeze()
            sub_nerf_tmp=torch.zeros((sub_nerf.shape[0],1),dtype = torch.float,device=sub_nerf.device)                 
            for i in range(self.hparams.sub_nerf_test_num):
                sub_nerf_tmp[sub_nerf_tmp > 0]=0#             
                sub_nerf_tmp[torch.logical_and(sub_nerf > (i+0.5), sub_nerf < (i+1.5))]=1 #    
                count_sub_nerf=torch.sum(sub_nerf_tmp)#    
                sub_nerf_mask = sub_nerf_tmp.bool().squeeze()
                pred_ranges_fine_sub = pred_ranges_fine[sub_nerf_mask]#
                pred_ranges_sub = pred_ranges[sub_nerf_mask]
                gt_ranges_sub = gt_ranges[sub_nerf_mask]      
                if(count_sub_nerf>=1):
                    loss_range=loss_range + 1e-1*self.hparams.lambda_loss*self.loss(1e1*pred_ranges_sub, 1e1*gt_ranges_sub)               
                    loss_range_fine=loss_range_fine+1e-1*self.hparams.lambda_loss_fine*self.loss(1e1*pred_ranges_fine_sub, 1e1*gt_ranges_sub)                             
        else:        
            loss_range =   1e-1*self.hparams.lambda_loss*self.loss(1e1*pred_ranges, 1e1*gt_ranges)                              
            loss_range_fine =   1e-1*self.hparams.lambda_loss*self.loss(1e1*pred_ranges_fine, 1e1*gt_ranges)              

        child_free_loss_fine=results['child_free_loss_fine']
        child_depth_loss_fine=results['child_depth_loss_fine']     
        child_free_loss=results['child_free_loss']
        child_depth_loss=results['child_depth_loss']
        
        loss = loss_range+loss_range_fine+ \
            self.hparams.lambda_child_free_loss*child_free_loss_fine+self.hparams.lambda_child_free_loss*child_free_loss+\
            self.hparams.lambda_child_depth_loss*child_depth_loss_fine+self.hparams.lambda_child_depth_loss*child_depth_loss    # 0511 新增                
        self.log('train/loss', loss)

        with torch.no_grad():
            abs_error_ = abs_error(pred_ranges, gt_ranges)
            acc_thres_ = acc_thres(pred_ranges, gt_ranges)# 
            self.log('train/avg_error', abs_error_)
            self.log('train/acc_thres', acc_thres_)
           
        if (self.hparams.current_epoch!=0) or (batch_idx>=20): #                  
            if (batch_idx%5==0):  # 
                if((self.hparams.current_epoch==0)):                    
                    self.hparams.current_epoch=self.hparams.current_epoch+1        
                    self.plotx.append(1)
                else:            
                    self.plotx.append(len(self.plotx)+1)
                self.ploty.append(loss.cpu().detach().numpy()) 

                loss_child_free = self.hparams.lambda_child_free_loss*child_free_loss 
                loss_child_free_fine = self.hparams.lambda_child_free_loss*child_free_loss_fine          
                loss_child_depth = self.hparams.lambda_child_depth_loss*child_depth_loss   #  
                loss_child_depth_fine = self.hparams.lambda_child_depth_loss*child_depth_loss_fine     
                self.plot_loss_range.append(loss_range.cpu().detach().numpy())         
                self.plot_loss_range_fine.append(loss_range_fine.cpu().detach().numpy())   
                self.plot_loss_child_free.append(loss_child_free.cpu().detach().numpy())         
                self.plot_loss_child_free_fine.append(loss_child_free_fine.cpu().detach().numpy()) 
                self.plot_loss_child_depth.append(loss_child_depth.cpu().detach().numpy())         #  
                self.plot_loss_child_depth_fine.append(loss_child_depth_fine.cpu().detach().numpy())   
                np.save(self.hparams.saveploty_path,arr=self.ploty)
                np.save(self.hparams.saveploty_path_range,arr=self.plot_loss_range)
                np.save(self.hparams.saveploty_path_range_fine,arr=self.plot_loss_range_fine)            
                np.save(self.hparams.saveploty_path_child_free,arr=self.plot_loss_child_free)
                np.save(self.hparams.saveploty_path_child_free_fine,arr=self.plot_loss_child_free_fine)                     
                np.save(self.hparams.saveploty_path_child_depth,arr=self.plot_loss_child_depth) #  
                np.save(self.hparams.saveploty_path_child_depth_fine,arr=self.plot_loss_child_depth_fine)    

            if(self.hparams.visualize==1) :
                plt.cla()
                plt.title("train/loss")
                plt.plot(self.plotx, self.ploty,label='loss')
                plt.plot(self.plotx, self.plot_loss_range,label='loss_range')
                plt.plot(self.plotx, self.plot_loss_range_fine,label='loss_range_fine')                        
                plt.plot(self.plotx, self.plot_loss_child_free,label='loss_child_free')
                plt.plot(self.plotx, self.plot_loss_child_free_fine,label='loss_child_free_fine')      
                plt.plot(self.plotx, self.plot_loss_child_depth,label='loss_child_depth')  #  
                plt.plot(self.plotx, self.plot_loss_child_depth_fine,label='loss_child_depth_fine')      
                plt.xlabel("iterations/5")
                plt.ylabel("train/loss")
                plt.legend(loc='best')            
                plt.pause(0.02)

        return loss

    def validation_step(self, batch, batch_idx):
        rays, gt_ranges = decode_batch(batch)
        rays = rays.squeeze()  # shape: (N_beams, 6)
        gt_ranges = gt_ranges.squeeze()  # shape: (N_beams,)
        results = self.forward(rays, True)#        
        pred_ranges = results['depth_fine']       # 
        rays_o, rays_d = rays[:, :3], rays[:, 3:6]
        pred_pts = rays_o + rays_d * pred_ranges.unsqueeze(-1)
        gt_pts = rays_o + rays_d * gt_ranges.unsqueeze(-1)  
        valid_mask_gt=torch.ones((gt_ranges.shape[0],),dtype = torch.bool)               
        cd, fscore = eval_points(pred_pts, gt_pts, valid_mask_gt)

        loss=torch.tensor([0],device=rays.device)                   ## 
        abs_error_=torch.tensor([0],device=rays.device)                        
        acc_thres_=torch.tensor([0],device=rays.device)      
        if self.hparams.use_child_nerf_divide==1: # 
            sub_nerf= rays[:, 9].view(-1, 1)
            sub_nerf=sub_nerf.squeeze()
            sub_nerf_tmp=torch.zeros((sub_nerf.shape[0],1),dtype = torch.float,device=sub_nerf.device)                 
            count_tmp= torch.tensor([0],device=sub_nerf.device)            
            for i in range(self.hparams.sub_nerf_test_num):
                sub_nerf_tmp[sub_nerf_tmp > 0]=0#             
                sub_nerf_tmp[torch.logical_and(sub_nerf > (i+0.5), sub_nerf < (i+1.5))]=1 #    
                count_sub_nerf=torch.sum(sub_nerf_tmp)#    
                sub_nerf_mask = sub_nerf_tmp.bool().squeeze()
                pred_ranges_sub = pred_ranges[sub_nerf_mask]
                gt_ranges_sub = gt_ranges[sub_nerf_mask]            
                valid_mask_gt_sub=torch.ones((gt_ranges_sub.shape[0],),dtype = torch.bool)                         
                                
                if(count_sub_nerf>=1):
                    loss_sub = self.loss(pred_ranges_sub, gt_ranges_sub, valid_mask_gt_sub)
                    abs_error_sub = abs_error(pred_ranges_sub, gt_ranges_sub, valid_mask_gt_sub)
                    acc_thres_sub = acc_thres(pred_ranges_sub, gt_ranges_sub, valid_mask_gt_sub)
                    loss=loss+loss_sub
                    abs_error_=abs_error_+abs_error_sub
                    acc_thres_=acc_thres_+acc_thres_sub              
                    count_tmp=count_tmp+1  
            loss=loss/count_tmp
            abs_error_=abs_error_/count_tmp
            acc_thres_=acc_thres_/count_tmp 
        else:
            valid_mask_gt=torch.ones((gt_ranges.shape[0],),dtype = torch.bool)                                     
            loss = self.loss(pred_ranges, gt_ranges, valid_mask_gt)
            abs_error_ = abs_error(pred_ranges, gt_ranges, valid_mask_gt)
            acc_thres_ = acc_thres(pred_ranges, gt_ranges, valid_mask_gt)
       
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/avg_error', abs_error_, prog_bar=True)
        self.log('val/acc_thres', acc_thres_, prog_bar=True)
        self.log('val/cd', cd, prog_bar=True)
        self.log('val/fscore', fscore, prog_bar=True)


if __name__ == '__main__':
    print("started training=========================================================================")
    print("started training=========================================================================")
    print("started training=========================================================================")        
    hparams = get_opts()

    torch.set_float32_matmul_precision('high')

    if hparams.seed:
        seed_everything(hparams.seed, workers=True)

    print("hparams:\n",hparams)
    nof_system = NOFSystem(hparams=hparams)

    checkpoint_callback = ModelCheckpoint(
        monitor='train/loss', mode='min', save_top_k=5, filename='best', save_last=True)    

    logger = TensorBoardLogger(
        save_dir="logs",
        name=hparams.exp_name,
    )

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=1,
                      num_sanity_val_steps=-1,
                      benchmark=True,
                      log_every_n_steps=1,  #  
                    )     # 
    trainer.fit(nof_system)

    plt.ioff()
    plt.show()
    print(checkpoint_callback.best_model_path)
