import math

from lib.models.memtrack_seq import build_memtrack_seq
from lib.test.tracker.basetracker import BaseTracker
import torch
import numpy as np
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d,gauss_1d,hann2d_bias,gaussian_window
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
# for debug
import cv2
import os
import torch.nn.functional as F
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import matplotlib.pyplot as plt
from lib.utils.box_ops import box_xywh_to_cxcywh,box_cxcywh_to_xywh
from skimage.measure import label, regionprops


class MemTrackSeq(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MemTrackSeq, self).__init__(params)
        network = build_memtrack_seq(params.cfg, training=False) # the whole model
        print(self.params.checkpoint)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        
        self.cfg = params.cfg
        self.bins = self.cfg.MODEL.BINS
        self.range = self.cfg.MODEL.RANGE
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()  # preprocessing image
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        # self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = 0
        self.frame_id = 0   # initialize frame id
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.store_result = None
        self.save_all = self.cfg.MODEL.SEQNUMS
        self.x_feat = None
        self.update = None
        self.gauss_window = None
        self.window_lr = self.cfg.TEST.WINDOW_LR
        self.tp_num = self.cfg.TEST.TP_NUM

    def target_mask(self,bbox):
 
        mask = torch.zeros(self.params.template_size,self.params.template_size)
        bbox = bbox.to(torch.int32).tolist()
        x1,y1 = bbox[0],bbox[1]
        x2,y2 = bbox[2],bbox[3]
        mask[y1:y2,x1:x2] = 1   # utils.save_image(out,'mask.jpg')
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0),(self.params.template_size//self.cfg.MODEL.BACKBONE.STRIDE))
        return mask
    
    def initialize(self, image, info: dict):
        # forward the template once

        z_patch_arr, z_resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        
        z_bbox_wh = torch.tensor(info['init_bbox'][2:]) * z_resize_factor 
        z_bbox_cxcy = torch.tensor((self.params.template_size/2,self.params.template_size/2))
        z_bbox = torch.cat([z_bbox_cxcy-z_bbox_wh/2,z_bbox_cxcy+z_bbox_wh/2],dim=0)
        template_label = self.target_mask(z_bbox)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
 
        with torch.no_grad():
            self.z_dict1 = template
            self.target_label = template_label.to(template.tensors)

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], z_resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)
        # save states
        self.state = info['init_bbox']
        trajectory = np.array(info['init_bbox'][:2]) + np.array(info['init_bbox'][2:]) / 2
        self.trajectory = trajectory.reshape(-1,2)
        self.frame_id = 0
        # self.store_result = [info['init_bbox'].copy()]
        search_bbox_wh = torch.tensor(info['init_bbox'][2:]) *self.params.search_size / ((self.params.template_size / z_resize_factor) * (self.params.search_factor / self.params.template_factor)) 
        w, h = search_bbox_wh[0] / self.cfg.MODEL.BACKBONE.STRIDE, search_bbox_wh[1] / self.cfg.MODEL.BACKBONE.STRIDE
        if w <= h:
            ratio = w / h
            # h = math.exp(ratio-1) * h
            w = (math.log10(1/ratio) * 2 + 1) * w
        else:
            ratio = h / w
            # w = math.exp(ratio-1) * w
            h = (math.log10(1/ratio) * 2 + 1) * h
        self.output_window = gaussian_window((self.feat_sz, self.feat_sz),sigma_x=w,sigma_y=h).cuda().unsqueeze(0).unsqueeze(1)
        # plt.imshow(self.output_window.clone().cpu().numpy().squeeze(), cmap='viridis')
        # plt.savefig('gauss.jpg')

        self.prev_area = torch.prod(search_bbox_wh).reshape(-1)
        self.store_wh = torch.tensor(info['init_bbox'][2:]).reshape(-1,2)
   
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_mask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_mask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.track(frame_id=self.frame_id,template=self.z_dict1.tensors,
                                            search=x_dict.tensors, target_label=self.target_label,
                                            ce_template_mask=self.box_mask_z, ce_keep_rate=None)
        pred_boxes = out_dict['bbox']
        pred_score_map = out_dict['score_map']
        size_map = out_dict['size_map']
        offset_map = out_dict['offset_map']
        pred_label = out_dict['pred_label']
        backbone_feat = out_dict['backbone_feat']

        heatmap = out_dict['score_ve'].reshape(1,1,int(out_dict['score_ve'].shape[-1]**.5),-1)
        heatmap = F.interpolate(heatmap,self.params.search_size,mode='bilinear',align_corners=True)
        heatmap = heatmap.flatten()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = heatmap.reshape(self.params.search_size,-1).cpu().numpy()

        response = pred_score_map * self.output_window
 
        # update memory 
        update,use_network = self.determine_updatemMemory(pred_label,resize_factor,heatmap,response)
        if update:
            memory = self.network.memory_network(mode='encode',feat=backbone_feat,mask=pred_label)
            self.network.memory_network(mode='addMemory',score_ve=out_dict['score_ve'],in_memorys=memory) 
        '''one order polyminoal function'''
        if self.frame_id > self.tp_num and not use_network: 
            x = self.trajectory[:,0][-self.tp_num:]
            y = self.trajectory[:,1][-self.tp_num:]
            frame_num = np.arange(0,self.tp_num)
            func_x = np.polyfit(frame_num, x, 1)
            func_y = np.polyfit(frame_num, y, 1)
            pred_x = np.polyval(func_x,self.tp_num + 1)
            pred_y = np.polyval(func_y,self.tp_num + 1)
            w,h = self.store_wh[:,0].mean().item(),self.store_wh[:,1].mean().item()
            self.state = clip_box([pred_x-w/2,pred_y-h/2,w,h], H, W, margin=2)
        # '''two order polyminoal function'''
        # if self.frame_id > self.tp_num  and not use_network: 
        #     x = self.trajectory[:,0][-self.tp_num :]
        #     y = self.trajectory[:,1][-self.tp_num :]
        #     frame_num = np.arange(0,self.tp_num )
        #     func_x = np.polyfit(frame_num, x, 2)
        #     func_y = np.polyfit(frame_num, y, 2)
        #     pred_x = np.polyval(func_x,self.tp_num  + 1)
        #     pred_y = np.polyval(func_y,self.tp_num  + 1)
        #     w,h = self.store_wh[:,0].mean().item(),self.store_wh[:,1].mean().item()
        #     self.state = clip_box([pred_x-w/2,pred_y-h/2,w,h], H, W, margin=2)
        elif not use_network:
            v_x = (self.trajectory[-1,0] - self.trajectory[0,0]) / len(self.trajectory)
            v_y = (self.trajectory[-1,1] - self.trajectory[0,1]) / len(self.trajectory)

            pred_x = v_x + self.trajectory[-1,0]
            pred_y = v_y + self.trajectory[-1,1]
            w,h = self.store_wh[:,0].mean().item(),self.store_wh[:,1].mean().item()
            self.state = clip_box([pred_x-w/2,pred_y-h/2,w,h], H, W, margin=2)
        else:
            # add hann windows
            pred_boxes = self.network.box_head.cal_bbox(response, size_map, offset_map)
            pred_boxes = pred_boxes.view(-1, 4)
            # corresponde search region bbox
            search_bbox = ((pred_boxes.mean(dim=0)) * self.params.search_size).cpu()
            search_area = torch.prod(search_bbox[2:]).reshape(-1)
            
            self.prev_area = torch.cat([self.prev_area,search_area],dim=0)
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=2)

        store_wh = torch.tensor([self.state[2],self.state[3]]).reshape(-1,2)
        self.store_wh = torch.cat([self.store_wh,store_wh],dim=0)

        cx,cy = self.state[0] + self.state[2]/2,self.state[1] + self.state[3]/2
        self.trajectory = np.concatenate([self.trajectory,np.array([cx,cy]).reshape(1,-1)])
        # for debug
        if self.debug:
            if not self.use_visdom:
                frame_id = self.frame_id
                x1, y1, w, h = self.state
                gt_x1,gt_y1,gt_w,gt_h = info['gt_bbox']
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=1)
                cv2.rectangle(image_BGR, (int(gt_x1),int(gt_y1)), (int(gt_x1+gt_w),int(gt_y1+gt_h)), color=(0,215,255), thickness=1)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % frame_id)
                cv2.imwrite(save_path, image_BGR)
                pred_label = F.interpolate(pred_label,self.params.search_size)
                cv2.imwrite(os.path.join(self.save_dir, "%04dx.jpg" % frame_id),x_patch_arr*pred_label.squeeze(0).permute(1,2,0).cpu().numpy())
                cv2.imwrite(os.path.join(self.save_dir, "%04dx_oir.jpg" % frame_id),x_patch_arr)
                score_ve = out_dict['score_ve'].reshape(1,1,int(out_dict['score_ve'].shape[-1]**.5),-1)
                score_ve = F.interpolate(score_ve,self.params.search_size,mode='bilinear',align_corners=True)
                response = F.interpolate(response,self.params.search_size,mode='bilinear',align_corners=True)
                _,_,H,W = score_ve.shape
                score_ve = score_ve.flatten()
                score_ve = (score_ve - score_ve.min()) / (score_ve.max() - score_ve.min())
                score_ve = score_ve.reshape(H,W).cpu().numpy()
                response = response.flatten()
                response = (response - response.min()) / (response.max() - response.min())
                response = response.reshape(H,W).cpu().numpy()
                # 将热力图从0-1映射到0-255并应用颜色映射
                heatmap = np.uint8(255 * score_ve)
                response_heatmap =  np.uint8(255 * response)
                colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 使用J色表，获得伪彩色热力图
                response_colored_heatmap = cv2.applyColorMap(response_heatmap, cv2.COLORMAP_JET)
                # 叠加热力图和搜索图像（50%透明度）
                show_heatmap = cv2.addWeighted(colored_heatmap, 0.5,x_patch_arr, 0.5, 0)
                show_response_heatmap = cv2.addWeighted(response_colored_heatmap, 0.3,x_patch_arr, 0.7, 0)
                cv2.putText(show_response_heatmap, '{}'.format(use_network), (300, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 4,)
                cv2.imwrite(os.path.join(self.save_dir, "%04dheatmap.jpg" % frame_id),show_heatmap)
                cv2.imwrite(os.path.join(self.save_dir, "%04dresponse_heatmap.jpg" % frame_id),show_response_heatmap)
                inp_box_feat = out_dict['heatmap'].mean(dim=1,keepdim=True) * (-1)
                inp_box_feat = torch.exp(inp_box_feat*10)
                inp_box_feat = F.interpolate(inp_box_feat,self.params.search_size,mode='bilinear',align_corners=True)
                inp_box_feat = inp_box_feat - inp_box_feat.mean()
                inp_box_feat = inp_box_feat.clamp(0,inp_box_feat.max())
                inp_box_feat = (inp_box_feat - inp_box_feat.min()) / (inp_box_feat.max() - inp_box_feat.min())
                inp_box_feat = inp_box_feat.reshape(self.params.search_size,-1).cpu().detach().numpy()
                inp_box_feat =  np.uint8(255 * inp_box_feat)
                inp_box_feat_colored_heatmap = cv2.applyColorMap(inp_box_feat, cv2.COLORMAP_JET)
                show_inp_box_feat = cv2.addWeighted(inp_box_feat_colored_heatmap, 0.4,x_patch_arr, 1, 0)
                cv2.imwrite(os.path.join(self.save_dir, "%04dinp_box_feat.jpg" % frame_id),show_inp_box_feat)

            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')
                if out_dict['update_pred_label'] is not None:
                    pred_label = out_dict['update_pred_label'].squeeze(0).cpu().int()
                    dynamic_img = torch.from_numpy(x_patch_arr).permute(2, 0, 1) * pred_label
                    self.visdom.register(dynamic_img, 'image', 1, 'dynamic_region')
                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                # self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break
        bias = None
        if len(self.trajectory) > 5:
            bias_x = (self.trajectory[-1,0] - self.trajectory[-5,0])  * resize_factor / self.cfg.MODEL.BACKBONE.STRIDE / 4
            bias_y = (self.trajectory[-1,1] - self.trajectory[-5,1])  * resize_factor / self.cfg.MODEL.BACKBONE.STRIDE / 4
            bias = (bias_x,bias_y)
        w,h = self.state[2] * resize_factor  / self.cfg.MODEL.BACKBONE.STRIDE, self.state[3]* resize_factor  / self.cfg.MODEL.BACKBONE.STRIDE
        if w <= h:
            ratio = w / h
            w = (math.log10(1/ratio) * 2 + 1) * w
        else:
            ratio = h / w
            h = (math.log10(1/ratio) * 2 + 1) * h

        self.output_window = gaussian_window([self.feat_sz, self.feat_sz],bias=bias,sigma_x=w,sigma_y=h).cuda()
        # show_hann = self.output_window.cpu().clone().squeeze().numpy()
        # plt.imshow(show_hann, cmap='viridis')
        # plt.savefig('gauss.jpg')

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}
        
    def peak_num(self,hp_score):

        high_value_group = hp_score>(np.max(hp_score)-0.1)
        
        label_image = label(high_value_group)

        regions = regionprops(label_image)
        
        high_value_centers = [tuple(map(int, region.centroid)) for region in regions]
        return(high_value_centers)
    
    def determine_updatemMemory(self,pred_label,resize_factor,heatmap,response):
        prev_w,prev_h = self.state[2],self.state[3]
        prev_area = self.prev_area.mean() 
        label = F.interpolate(pred_label.to(torch.float32),self.params.search_size).cpu()
        pred_label_area = label.sum()
        pred_cy,pred_cx = (torch.where(label==1)[2].unique().to(torch.float32)+1e-4).mean(),(torch.where(label==1)[3].unique().to(torch.float32)+ 1e-4).mean()
        prev_cx = prev_cy = self.params.search_size /2

        response = F.interpolate(response.clone().cpu(),self.params.search_size,mode='bilinear',align_corners=True).squeeze()
        response = (response - response.min()) / (response.max() - response.min())
        response_cy, response_cx = torch.where(response==response.max())
        response_cy, response_cx = response_cy.float().mean(), response_cx.float().mean()
        
        
        high_val_center = self.peak_num(heatmap)

        if len(high_val_center) == 1:
            heatmap_cx = high_val_center[0][1]
            heatmap_cy = high_val_center[0][0]

        else:
            return False,False
        if prev_w <= prev_h:
            ratio = torch.tensor((prev_w/ prev_h))
            threshold_h = torch.exp(ratio-1) * self.store_wh[:,1].mean()
            isin_w = abs(pred_cx - prev_cx) / resize_factor< self.store_wh[:,0].mean()
            isin_h = abs(pred_cy - prev_cy) / resize_factor< threshold_h
            response_isin_w = abs(response_cx - prev_cx) / resize_factor< self.store_wh[:,0].mean()
            response_isin_h = abs(response_cy - prev_cy) / resize_factor< threshold_h
        else:
            ratio = torch.tensor((prev_h / prev_w))
            threshold_w = torch.exp(ratio-1) * self.store_wh[:,0].mean()
            isin_w = abs(pred_cx - prev_cx) / resize_factor < threshold_w
            isin_h = abs(pred_cy - prev_cy) / resize_factor < self.store_wh[:,1].mean()
            response_isin_w = abs(response_cx - prev_cx) / resize_factor< threshold_w
            response_isin_h = abs(response_cy - prev_cy) / resize_factor< self.store_wh[:,1].mean()
       
        if (pred_label_area > 0.7 *prev_area and pred_label_area < 1.2*prev_area) and (isin_w and isin_h) and (response_isin_w and response_isin_h):
            return True,True
        elif (response_isin_w and response_isin_h): 
            return False,True
        else:
            return False,False
        
    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights

def get_tracker_class():
    return MemTrackSeq
