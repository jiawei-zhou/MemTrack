from collections import OrderedDict
from einops import rearrange
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image as PILImage
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import cv2
try:
    import wandb
except ImportError:
    raise ImportError(
        'Please run "pip install wandb" to install wandb')


class WandbWriter:
    def __init__(self, exp_name, cfg, output_dir, cur_step=0, step_interval=0):
        self.wandb = wandb
        self.step = cur_step
        self.interval = step_interval
        self.mean = torch.tensor((0.485, 0.456, 0.406))
        self.std = torch.tensor((0.229, 0.224, 0.225))
        self.epoch = 1
        wandb.init(project="HisTrack", name=exp_name, config=cfg, dir=output_dir)

    def write_log(self, stats: OrderedDict, epoch=-1,data=None):
        self.step += 1
        for loader_name, loader_stats in stats.items():
            if loader_stats is None:
                continue

            log_dict = {}
            for var_name, val in loader_stats.items():
                if hasattr(val, 'avg'):
                    log_dict.update({loader_name + '/' + var_name: val.avg})
                else:
                    log_dict.update({loader_name + '/' + var_name: val.val})

                if epoch >= 0:
                    log_dict.update({loader_name + '/epoch': epoch})

            self.wandb.log(log_dict, step=self.step*self.interval)

            if data is not None and epoch - self.epoch:
            # if data is not None:
                search_imgs = rearrange(data['search_images'].detach().cpu(),'n b c h w -> (b n) h w c')
                pred_label = rearrange(F.interpolate(data['pred_label'],search_imgs.shape[1]),'b c h w -> b h w c')
                score_ves = rearrange(F.interpolate(data['score_ve'],search_imgs.shape[1],mode='bilinear',align_corners=True),'b c h w -> b h (w c)') # b h w
                B,H,W = score_ves.shape
                score_ves = rearrange(score_ves,'b h w -> b (h w)')
                score_ves = (score_ves - score_ves.min(dim=-1,keepdim=True)[0]) / (score_ves.max(dim=-1,keepdim=True)[0] - score_ves.min(dim=-1,keepdim=True)[0])
                score_ves = rearrange(score_ves,'b (h w) -> b h w',h=H,w=W)
                mean,std = self.mean.reshape(1,1,1,3).to(search_imgs), self.std.reshape(1,1,1,3).to(search_imgs) 
                search_imgs = (search_imgs * std + mean) 
                pad = (0,1,0,1)
                search_imgs = search_imgs * 255 # b h w c

                mask_imgs = search_imgs * pred_label
                mask_imgs = rearrange(mask_imgs,'n h w c -> n c h w')
                mask_imgs = F.pad(mask_imgs,pad,mode='constant',value=1)
                mask_imgs = rearrange(mask_imgs,'n c h w -> n h w c')
                mask_imgs = mask_imgs.numpy().astype(np.uint8)
                for i in range(len(mask_imgs)):
                    if i == 0:
                        show_img = mask_imgs[i]
                        # 将热力图从0-1映射到0-255并应用颜色映射
                        heatmap = np.uint8(255 * score_ves[i])
                        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 使用J色表，获得伪彩色热力图
                        # 叠加热力图和搜索图像（50%透明度）
                        show_heatmap = cv2.addWeighted(colored_heatmap, 0.5, np.array(search_imgs[i]).astype(np.uint8), 0.5, 0)
                    else:
                        show_img = np.concatenate([show_img,mask_imgs[i]],axis=1)
                        heatmap = np.uint8(255 * score_ves[i])
                        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 使用J色表，获得伪彩色热力图
                        # 叠加热力图和搜索图像（50%透明度）
                        overlay_image = cv2.addWeighted(colored_heatmap, 0.5, np.array(search_imgs[i]).astype(np.uint8), 0.5, 0)
                        show_heatmap = np.concatenate([show_heatmap,overlay_image],axis=1)

                Img = wandb.Image(show_img, caption="pred_label")
                show_heatmap = wandb.Image(show_heatmap, caption="score_ve_img")
                self.wandb.log({"pred_label":Img}, step=self.step*self.interval)
                self.wandb.log({"score_ve_img":show_heatmap}, step=self.step*self.interval)
                self.epoch = epoch