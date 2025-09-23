"""
Basic OSTrack model.
"""
import os
from collections import OrderedDict
import torch
from torch import nn
import cv2
from lib.models.layers.box_head import build_box_head
from lib.models.layers.seek_target import build_seek_target
from lib.models.memtrack.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.models.memtrack_seq.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.models.layers.memory import build_memory_network
import torch.nn.functional as F
from einops import rearrange
from torchvision import utils
import matplotlib.pyplot as plt
class MemTrackSeq(nn.Module):
    """ This is the base class for ARTrack """

    def __init__(self, backbone, pix_head,memory_network,seek_tgt_block,tz,stride):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
        """
        super().__init__()
        self.backbone = backbone
        self.box_head = pix_head
        self.memory_network = memory_network
        self.seek_tgt_block = seek_tgt_block
        z_len = (tz // stride) ** 2
        self.z_len = z_len
        self.template_center = [z_len // 2 -1,z_len//2 +1]
        
    def forward(self,mode='forward',frame_id=None,template=None,search=None,template_pixle_label=None,ce_template_mask=None,
                ce_keep_rate=None,template_label=None,search_label=None,search_anno= None,
                update_index=None,replace_index=None,feat=None,mask=None,score_ve=None,
                in_memorys=None,response=None,size_map=None,offset_map=None,
                ):
        """## only trained the N-1 nums search frames

        ### Args:
            - `template (torch.Tensor)`: b,c,h,w
            - `search (torch.Tensor)`: n,b,c,h,w
            - `history_seq (torch.Tensor)`: b,prenum*4

        ### Returns:
            - `_type_`: _description_
        """
        if mode == 'track':
            out = self.track(frame_id,template,search,template_pixle_label,ce_template_mask,ce_keep_rate)
            return out
        if mode == 'Memencode':
            out = self.memory_network(mode='encode',feat=feat,mask=mask)
            return out
        if mode == 'getMemory':
            out = self.memory_network.memory
            return out
        if mode == 'addMemory':
            self.memory_network(mode='addMemory',score_ve=score_ve,in_memorys=in_memorys,state='train',update_list=update_index,replace_index=replace_index)
            return None
        if mode == 'getbbox':
            out = self.box_head.cal_bbox(response, size_map, offset_map)
            return out
        if mode == 'profile':
            out = self.track_profile(frame_id,template,search,template_pixle_label,ce_template_mask,ce_keep_rate)
            return out
        B= search.shape[0]
        self.bs = B

        z,x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,template_pixle_lable=template_pixle_label,
                                    return_last_attn=False)
 
        z = rearrange(z,' b (h w) c -> b c h w',h=int(z.shape[1]**.5))[0:1]
        x = rearrange(x,'b (h w) c -> b c h w',h=int(x.shape[1]**.5))
 
        x_pos = self.backbone.pos_embed_x

        _,C,H,W = x.shape
        x_feat_list = []
        
        for i in range(B):
            if i == 0:
                memory_all = self.memory_network(mode='encode',feat=z,mask=template_label,state='initial')  
                memory_all = memory_all.unsqueeze(0)    # n b c h w

            else:
                if update_index[i-1]:
                    if len(memory_all) < self.memory_network.memory_num:
                        memory = self.memory_network(mode='encode',feat=x[i-1:i],mask=search_label[i-1:i])
                        memory_all = torch.cat([memory_all,memory.unsqueeze(0)])
                    else:
                        index = replace_index[i-1]
                        temp = torch.ones_like(memory_all) # n b c h w
                        temp[index] = temp[index] - 1
                        memory = self.memory_network(mode='encode',feat=x[i-1:i],mask=search_label[i-1:i])
                        memory_all = torch.where(temp.to(torch.bool),memory_all,memory.unsqueeze(0))

            x_feat = self.memory_network(mode='decode',x_feat=x[i:i+1],x_pos=x_pos,memory=memory_all)
            x_feat_list.append(x_feat)

        x = torch.stack(x_feat_list,dim=0)
        x = rearrange(x,'n b c h w -> (n b) c h w')
        out = self.forward_head(x)
        out['score_ve'] = aux_dict['score_ve']
        return out

    def forward_head(self, response_map):
         # run the center head
        score_map_ctr, bbox, size_map, offset_map = self.box_head(response_map)
        # outputs_coord = box_xyxy_to_cxcywh(bbox)
        outputs_coord = bbox
        out = {'bbox': outputs_coord,
                'score_map': score_map_ctr,
                'size_map': size_map,
                'offset_map': offset_map}
        out['pred_label_feat'] = self.seek_tgt_block(response_map)
        return out
    
    def track(self,frame_id,template,search,target_label,ce_template_mask,ce_keep_rate):
        z,x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,template_pixle_lable=target_label,
                                    return_last_attn=False)
        z = rearrange(z,'b (h w) c -> b c h w',h=int(z.shape[1]**.5))
        x = rearrange(x,'b (h w) c -> b c h w',h=int(x.shape[1]**.5))
        x_pos = self.backbone.pos_embed_x
        if frame_id == 1:
            x_feat = self.memory_network(mode='initialize',z_feat=z,mask=target_label,x_feat=x,x_pos=x_pos)
        else:
            x_feat = self.memory_network(mode='decode',x_feat=x,x_pos=x_pos)
        out = self.forward_head(x_feat)
        out['backbone_feat'] = x
        out['score_ve'] = aux_dict['score_ve']
        out['ce_mask'] = aux_dict['removed_indexes_s']
        out['heatmap'] = x_feat
        pred_label_feat = out['pred_label_feat']

        out['pred_label'] = torch.where(pred_label_feat.softmax(dim=1)[:,[1]]>0.6,1,0).to(torch.float32)
        return out
    
    def track_profile(self,frame_id,template,search,target_label,ce_template_mask,ce_keep_rate):

        z,x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,template_pixle_lable=target_label,
                                    return_last_attn=False)
        z = rearrange(z,'b (h w) c -> b c h w',h=int(z.shape[1]**.5))
        x = rearrange(x,'b (h w) c -> b c h w',h=int(x.shape[1]**.5))
        x_pos = self.backbone.pos_embed_x
        if frame_id == 1:
            x_feat = self.memory_network(mode='initialize',z_feat=z,mask=target_label,x_feat=x,x_pos=x_pos)
        else:
            x_feat = self.memory_network(mode='decode',x_feat=x,x_pos=x_pos)
        out = self.forward_head(x_feat)
        out['backbone_feat'] = x
        out['score_ve'] = aux_dict['score_ve']
        out['ce_mask'] = aux_dict['removed_indexes_s']
        pred_label_feat = out['pred_label_feat']
  
        out['pred_label'] = torch.where(pred_label_feat.softmax(dim=1)[:,[1]]>0.6,1,0).to(torch.float32)
        memory = self.memory_network(mode='encode',feat=x,mask=out['pred_label'])
        self.memory_network(mode='addMemory',score_ve=out['score_ve'],in_memorys=memory) 

        return out
    
    def adjust_outdict(self,out):
        for k,v in out.items():
            if isinstance(v,torch.Tensor):
                if v.ndim == 4:
                    out[k] = rearrange(v,'(n b) c h w -> n b c h w',b=self.bs)
                elif v.ndim == 3:
                    out[k] = rearrange(v,'(n b) l c -> n b l c',b=self.bs)
                elif v.ndim == 2:
                    out[k] = rearrange(v,'(n b) c  -> n b c',b=self.bs)
        return out
    
    def visiual_mask_imgs(self,img,label,mask_img):
        sv_path = 'vis_mask_imgs/'
        utils.save_image(img,sv_path+'oir.jpg')
        utils.save_image(label,sv_path+'mask.jpg')
        utils.save_image(mask_img,sv_path+'mask_img.jpg')

    def update_dict(self,dicts,data):
        if dicts == {}:
            for key in list(data.keys()):
                dicts[key] = []
                dicts[key].append(data[key])
        else:
            for key in list(dicts.keys()):
                dicts[key].append(data[key])
        return dicts
    
    def apply_stack(self,dicts):
        for key in list(dicts.keys()):
            if isinstance(dicts[key][0],torch.Tensor):
                dicts[key] = torch.stack(dicts[key],dim=0)
            else:
                dicts[key] = dicts[key][-1]
        return dicts
    

    def fill_label(self,lable_index,type='dyn'):
        """_summary_

        Args:
            lable_index (tensor): dynamic feat label (b,1,h,w)
        """
        lable_index = lable_index.permute(0,2,3,1)
        if type == 'temp':
            lable_index = lable_index.repeat(1,1,1,3)
            lable = torch.where(lable_index==1,self.templabel_fd,self.templabel_bd)
        else:
            lable_index = lable_index
            lable = torch.where(lable_index>=0.6826,self.dynlabel_fd,self.dynlable_bd)
        return lable.permute(0,3,1,2)
    
    def adjustment(self,label,bbox):
        """ adjust the valid range of target feat 

        Args:
            label (tensor): b,1,h,w
            bbox (tensor): (x1,y1,x2,y2)
        """
        bs = label.shape[0]
        label_mask = torch.zeros_like(label)
        update_index = []
        for i in range(bs):
            intersection = label[i,:,bbox[i,1]:bbox[i,3],bbox[i,0]:bbox[i,2]]   # 1,h,w
            intersection = torch.where(intersection>=0.5,1,0)
            if intersection.sum() == 0:
                update_index.append(torch.tensor((False)).to(intersection))
                continue
            pre_cx,pre_cy = torch.where(label[i]>=0.5)[2].to(torch.float32).mean(),torch.where(label[i]>=0.6826)[1].to(torch.float32).mean()
            pre_xy = torch.stack((pre_cx,pre_cy),dim=0)     
            box_cx,box_cy = (bbox[i,0] + bbox[i,2]) / 2 , (bbox[i,1] + bbox[i,3]) / 2 
            box_xy = torch.stack((box_cx,box_cy),dim=0)
            min_wh,max_wh = min((bbox[i,2] - bbox[i,0]),(bbox[i,3] - bbox[i,1])), max((bbox[i,2] - bbox[i,0]),(bbox[i,3] - bbox[i,1]))
            bound = torch.stack((min_wh,max_wh*0.6826),dim=0)
            update_index.append(self.pdist(pre_xy,box_xy) < torch.sqrt(torch.pow(bound,2).sum()))
            label_mask[i,:,bbox[i,1]:bbox[i,3],bbox[i,0]:bbox[i,2]] = intersection
        # label_mask = torch.where(label_mask>=0.6826,1,0)    # b,c,h,w
        update_index = torch.stack(update_index)
        return update_index,label_mask
    
    def updating_dfeat(self,index,d_feat,label):
        if (index==False).all():
            return self.d_feat
        else:
            dat_idx = torch.where(index==True)[0]
            nodat_idx = torch.where(index==False)[0]
            nodat_d_feat = self.d_feat[nodat_idx,:,:].view(-1,*self.d_feat.shape[1:])   # b,l,c
            ids_restore = torch.argsort(torch.cat((dat_idx,nodat_idx),dim=0))
            dat_d_feat = d_feat[dat_idx,:,:,:].view(-1,*d_feat.shape[1:])   # b,c,h,w
            dat_label = label[dat_idx,:,:,:].view(-1,*label.shape[1:])  # b,1,h,w
            hw = dat_d_feat.shape[-1]
            dat_label = F.interpolate(dat_label,(hw,hw))
            dat_label = self.fill_label(dat_label)
            dat_d_feat = torch.cat((dat_d_feat,dat_label),dim=1)
            dat_d_feat = self.side_backbone(dat_d_feat)
            B,C,H,W = dat_d_feat.shape
            dat_d_feat = dat_d_feat.permute(0,2,3,1).reshape(B,-1,C)
            d_feat = torch.cat((dat_d_feat,nodat_d_feat),dim=0) # b,l,c
            d_feat = torch.gather(d_feat,dim=0,index=ids_restore.reshape(-1,1,1).repeat(1,H*W,C))
            return d_feat
        
    def visual_motion_features(self,mf,search_anno=None):
        motion_feature = mf.clone().mean(1).detach().cpu().numpy()
        if search_anno is not None:
            bbox = search_anno.clone().detach().cpu()
            bbox = (bbox * motion_feature.shape[-1]).to(torch.int32).numpy()
            bbox = bbox.reshape(-1,4)
            for m_feat,box in zip(motion_feature,bbox):
                plt.figure(figsize=(10, 10))
                plt.imshow(m_feat)
                show_box(box,plt.gca())
                plt.savefig('m_feat.jpg')
        for m_feat in motion_feature:
            plt.figure(figsize=(10, 10))
            plt.imshow(m_feat)
            plt.savefig('m_feat.jpg') 
        plt.close()

def build_memtrack_seq(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('MemTrack' not in cfg.MODEL.PRETRAIN_FILE and 'DropTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        backbone_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        print("i use vit_large")
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        backbone_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        backbone_dim = backbone.embed_dim
        patch_start_index = 1
        
    else:
        raise NotImplementedError
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    # tz = cfg.DATA.TEMPLATE.SIZE
    # cfg.MODEL.ADJUST_LAYER.IN_CH = backbone_dim
    # cfg.MODEL.ADJUST_LAYER.UNI_CH = int(backbone_dim + backbone_dim / 2 + backbone_dim / 4)
    # cfg.MODEL.ADJUST_LAYER.STRIDE = cfg.MODEL.BACKBONE.STRIDE
    # backbone_adjust_layer = build_backbone_adjust_layer(cfg.MODEL.ADJUST_LAYER)
    box_head = build_box_head(cfg, backbone_dim)
    seek_tgt = build_seek_target(backbone_dim)
    # side_backbone = build_sidebackbone(pretrained=True,extra_chan=0)
    # sidebackbone_adjust_layer = build_sidebackbone_adjust_layer(64+128+256,hidden_dim)
    # motion_encoder = build_motion_encode(**lowercase_keys(cfg.MODEL.MOTIONMODEL))
    memory_network = build_memory_network(cfg,backbone_dim)
    # memory_block = build_memory_block(**lowercase_keys(cfg.MODEL.MEMORYBLOCK),tz=tz,stride=cfg.MODEL.BACKBONE.STRIDE)
    # down_dim = cfg.MODEL.MOTIONMODEL.DIM
    model = MemTrackSeq(backbone,box_head,memory_network,seek_tgt,
        cfg.DATA.TEMPLATE.SIZE,
        cfg.MODEL.BACKBONE.STRIDE,
    )

    if cfg.MODEL.PRETRAIN_PTH != "":
        load_from = cfg.MODEL.PRETRAIN_PTH
        checkpoint = torch.load(load_from, map_location="cpu")
        # load_weight_tune(model,checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=True)
        print('Load pretrained model from: ' + load_from)

    if ('MemTrackSeq' in cfg.MODEL.PRETRAIN_FILE)and training:
        pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model

def load_weight_tune(model,ckpt):

    new_dict = OrderedDict()
    for k, v in ckpt["net"].items():
        if 'pix_head' in k:
            if 'temp_fuse' in k:
                k = k.replace("pix_head","memory_block")
                new_dict[k] = v
            elif 'z_pos' in k or 'd_pos' in k:
                k = k.replace("pix_head.","")
                new_dict[k] = v
            else:
                new_dict[k] = v
        else:
            new_dict[k] = v
    missing_keys, unexpected_keys = model.load_state_dict(new_dict,strict=False)
    print("unexpected_keys:",unexpected_keys)

def draw_mask_image(imgs_,masks):
    imgs = imgs_.clone()
    mean = torch.tensor((0.485, 0.456, 0.406)).reshape(1,1,1,3).to(imgs)
    std = torch.tensor((0.229, 0.224, 0.225)).reshape(1,1,1,3).to(imgs)
    imgs = ((imgs.permute(0,2,3,1) *std + mean) * 255).to(torch.int32).contiguous()
    b,h,w,c = imgs.shape
    masks = masks.reshape(b,1,h,w).permute(0,2,3,1)
    id = 0
    for img, mask in zip(imgs,masks):
        id += 1
        img = img.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        img = img * mask
        cv2.imwrite('MemTrack/visual_maskimg/'+str(id)+'.jpg',img)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2], box[3]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=2))   


    