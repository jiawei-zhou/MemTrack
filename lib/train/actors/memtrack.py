from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import math
import numpy as np
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from torch.nn.functional import l1_loss
import cv2
import matplotlib.pyplot as plt
from einops import rearrange
from torchvision import utils
import time
import torch.nn.functional as F
from collections import OrderedDict
def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)
    
def generate_sa_simdr(joints):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    num_joints = 48
    image_size = [256, 256]
    simdr_split_ratio = 1.5625
    sigma = 6

    target_x1 = np.zeros((num_joints,
                              int(image_size[0] * simdr_split_ratio)),
                             dtype=np.float32)
    target_y1 = np.zeros((num_joints,
                              int(image_size[1] * simdr_split_ratio)),
                             dtype=np.float32)
    target_x2 = np.zeros((num_joints,
                              int(image_size[0] * simdr_split_ratio)),
                             dtype=np.float32)
    target_y2 = np.zeros((num_joints,
                              int(image_size[1] * simdr_split_ratio)),
                             dtype=np.float32)
    zero_4_begin = np.zeros((num_joints, 1), dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):

        mu_x1 = joints[joint_id][0]
        mu_y1 = joints[joint_id][1]
        mu_x2 = joints[joint_id][2]
        mu_y2 = joints[joint_id][3]

        x1 = np.arange(0, int(image_size[0] * simdr_split_ratio), 1, np.float32)
        y1 = np.arange(0, int(image_size[1] * simdr_split_ratio), 1, np.float32)
        x2 = np.arange(0, int(image_size[0] * simdr_split_ratio), 1, np.float32)
        y2 = np.arange(0, int(image_size[1] * simdr_split_ratio), 1, np.float32)

        target_x1[joint_id] = (np.exp(- ((x1 - mu_x1) ** 2) / (2 * sigma ** 2))) / (
                        sigma * np.sqrt(np.pi * 2))
        target_y1[joint_id] = (np.exp(- ((y1 - mu_y1) ** 2) / (2 * sigma ** 2))) / (
                        sigma * np.sqrt(np.pi * 2))
        target_x2[joint_id] = (np.exp(- ((x2 - mu_x2) ** 2) / (2 * sigma ** 2))) / (
                        sigma * np.sqrt(np.pi * 2))
        target_y2[joint_id] = (np.exp(- ((y2 - mu_y2) ** 2) / (2 * sigma ** 2))) / (
                        sigma * np.sqrt(np.pi * 2))
    return target_x1, target_y1, target_x2, target_y2

# angle cost
def SIoU_loss(test1, test2, theta=4):
    eps = 1e-7
    cx_pred = (test1[:, 0] + test1[:, 2]) / 2
    cy_pred = (test1[:, 1] + test1[:, 3]) / 2
    cx_gt = (test2[:, 0] + test2[:, 2]) / 2
    cy_gt = (test2[:, 1] + test2[:, 3]) / 2

    dist = ((cx_pred - cx_gt)**2 + (cy_pred - cy_gt)**2) ** 0.5
    ch = torch.max(cy_gt, cy_pred) - torch.min(cy_gt, cy_pred)
    x = ch / (dist + eps)

    angle = 1 - 2*torch.sin(torch.arcsin(x)-torch.pi/4)**2
    # distance cost
    xmin = torch.min(test1[:, 0], test2[:, 0])
    xmax = torch.max(test1[:, 2], test2[:, 2])
    ymin = torch.min(test1[:, 1], test2[:, 1])
    ymax = torch.max(test1[:, 3], test2[:, 3])
    cw = xmax - xmin
    ch = ymax - ymin
    px = ((cx_gt - cx_pred) / (cw+eps))**2
    py = ((cy_gt - cy_pred) / (ch+eps))**2
    gama = 2 - angle
    dis = (1 - torch.exp(-1 * gama * px)) + (1 - torch.exp(-1 * gama * py))

    #shape cost
    w_pred = test1[:, 2] - test1[:, 0]
    h_pred = test1[:, 3] - test1[:, 1]
    w_gt = test2[:, 2] - test2[:, 0]
    h_gt = test2[:, 3] - test2[:, 1]
    ww = torch.abs(w_pred - w_gt) / (torch.max(w_pred, w_gt) + eps)
    wh = torch.abs(h_gt - h_pred) / (torch.max(h_gt, h_pred) + eps)
    omega = (1 - torch.exp(-1 * wh)) ** theta + (1 - torch.exp(-1 * ww)) ** theta

    #IoU loss
    lt = torch.max(test1[..., :2], test2[..., :2])  # [B, rows, 2]
    rb = torch.min(test1[..., 2:], test2[..., 2:])  # [B, rows, 2]

    wh = fp16_clamp(rb - lt, min=0)
    overlap = wh[..., 0] * wh[..., 1]
    area1 = (test1[..., 2] - test1[..., 0]) * (
            test1[..., 3] - test1[..., 1])
    area2 = (test2[..., 2] - test2[..., 0]) * (
            test2[..., 3] - test2[..., 1])
    iou = overlap / (area1 + area2 - overlap)

    SIoU = 1 - iou + (omega + dis) / 2
    return SIoU, iou
    
def ciou(pred, target, eps=1e-7):
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    factor = 4 / math.pi**2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    # CIoU
    cious = ious - (rho2 / c2 + v**2 / (1 - ious + v))
    return cious, ious


class MemTrackActor(BaseActor):
    """ Actor for training ARTrack models """

    def __init__(self, net, objective, loss_weight, settings, bins, search_size,cfg=None,state='one_stage_train'):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.bins = bins
        self.range = self.cfg.MODEL.RANGE
        self.search_size = search_size
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.smooth_l1 = torch.nn.SmoothL1Loss(beta=0.1,reduction='mean')
        self.focal = None
        self.cls_loss = None
        self.mean = torch.tensor((0.485, 0.456, 0.406))
        self.std = torch.tensor((0.229, 0.224, 0.225))
        self.state = state
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)
        # compute video losses
        B = data['template_images'].shape[1]
        status_dict = OrderedDict()
        total_loss = 0
        
        # compute video loss
        for bs in range(B):
            cur_preddict,cur_gtdict = self.sample_videos_dict(out_dict,data,bs)
            loss, status = self.compute_losses(cur_preddict, cur_gtdict,B)
            total_loss += loss / B
            self.update_status_dict(status,status_dict,bs)
        
        pred_label = rearrange(out_dict['pred_label_feat'],'n b c h w -> (b n) c h w').softmax(dim=1)[:,[1]].clone().detach().cpu()
        pred_label = torch.where(pred_label>0.6,1,0).to(torch.float32)
        data['pred_label'] = pred_label
        data['score_ve'] = rearrange(out_dict['score_ve'],'n b (c h w) -> (b n) c h w',c=1,h=self.search_size//self.settings.stride).clone().detach().cpu()
        return total_loss, status_dict
        
    def update_status_dict(self,status,status_dict,bs):
        if bs == 0:
            status_dict.update(status)
        else:
            for k,v in status.items():
                status_dict[k] += v

    def sample_videos_dict(self,pred_dict,gt_dict,bs):
        gt_dict_names = ['search_anno','gt_mask','seq_output','gt_gauss_mask','score_ve_label']
        pred_dict_names = ['head_bbox','score_map','pred_label','pred_label_feat','tgt_feat','seq_bbox','score_ve','bbox']
        cur_preddict,cur_gtdict = {},{}
        for k,v in pred_dict.items():
            if k in pred_dict_names:
                cur_preddict[k] = v[:,bs:bs+1,...]
        for k,v in gt_dict.items():
            if k in gt_dict_names:
                cur_gtdict[k] = v[:,bs:bs+1,...]
        return cur_preddict,cur_gtdict
    
    def forward_pass(self, data):
        self.bs = data['template_images'].shape[1]
        template_img = data['template_images'].reshape(self.bs,*data['template_images'].shape[2:])
        search_img = data['search_images']
        nums = len(search_img)
        template_annos = rearrange(data['template_anno'],'n b c -> (n b) c')
        search_lable = data['gt_mask']
        template_lable = rearrange(data['template_lable'],'n b c h w -> (n b) c h w')
        template_pixle_lable = template_lable
        box_mask_z = None
        ce_keep_rate = None
        
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_img.shape[0], template_img.device,
                                            template_annos)

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
        box_mask_z = rearrange(box_mask_z.unsqueeze(0).repeat(nums,1,1),'n b lz -> (n b) lz')   

        out_dict = self.net(template=template_img,
                            search=search_img,
                            search_anno = data['search_anno'],
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            template_lable=template_lable,
                            template_pixle_lable=template_pixle_lable,
                            search_lable= search_lable,
                            state =self.state)
        return out_dict

    def compute_losses(self, pred_dict, gt_dict,total_batch,return_status=True):

        gt_bbox = rearrange(gt_dict['search_anno'],'n b c -> (n b) c')  # (Ns, batch, 4) (x1,y1,w,h) -> (Ns * batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = torch.stack(gt_gaussian_maps,dim=0)

        # Get boxes
        pred_boxes = pred_dict['bbox']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (n,b,4) --> (n*b,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0,max=1.0) # (Ns * batch, 4) (x1,y1,x2,y2)

        # compute giou and iou
        try:
            head_giou_loss, head_iou = SIoU_loss(pred_boxes_vec, gt_boxes_vec,4)  # (BN,4) (BN,4)
            head_giou_loss = head_giou_loss.mean()
        except:
            head_giou_loss, head_iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        head_l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        # compute location loss
        if 'score_map' in pred_dict:
            score_map = rearrange(pred_dict['score_map'],'n b c h w -> (n b) c h w')
            head_location_loss = self.objective['focal'](score_map, gt_gaussian_maps)
        else:
            head_location_loss = torch.tensor(0.0, device=l1_loss.device)
        # compute clssification loss
        if self.cls_loss is None:
            cls_weigth = torch.ones(2).to(gt_boxes_vec)
            self.cls_loss = torch.nn.CrossEntropyLoss(weight=cls_weigth, size_average=True).to(gt_boxes_vec)


        pred_label_feat = rearrange(pred_dict['pred_label_feat'],'n b c h w -> (n b h w) c')
        gt_lable_ce = rearrange(gt_dict['gt_mask'],'n b c h w -> (n b h w c)').to(torch.int64)
        cls_location_loss = self.cls_loss(pred_label_feat,gt_lable_ce)

        gt_lable_l1 = rearrange(gt_dict['gt_gauss_mask'],'n b c h w -> (n b) c h w')
        pred_label = rearrange(pred_dict['pred_label_feat'],'n b c h w -> (n b) c h w').softmax(dim=1)[:,[1]]
        cls_l1_loss = self.objective['l1'](pred_label,gt_lable_l1)
        
        # # compute score_ve loss
        score_ve_label = gt_dict['score_ve_label']
        score_ve = rearrange(pred_dict['score_ve'],'n b lx -> (n b) lx')
        score_ve = (score_ve - score_ve.min(dim=-1,keepdim=True)[0]) / (score_ve.max(dim=-1,keepdim=True)[0] - score_ve.min(dim=-1,keepdim=True)[0])
        # scroe_ve_location_loss = self.objective['focal'](scroe_ve, gt_gaussian_maps)
        score_ve_l1_loss = self.smooth_l1(score_ve,rearrange(score_ve_label,'n b c h w -> (n b) (c h w)'))
        loss = self.loss_weight['giou'] * (head_giou_loss) + self.loss_weight['l1'] * (head_l1_loss + cls_l1_loss + score_ve_l1_loss) + \
                self.loss_weight['focal'] * (head_location_loss + cls_location_loss) 
        
        if return_status:
            # status for log
            mean_iou = head_iou.detach().mean()
            status = {"Loss/total": loss.item() / total_batch,
                      "Loss/giou": head_giou_loss.item() / total_batch,
                      "Loss/head_location": head_location_loss.item() / total_batch,
                      "Loss/head_l1": head_l1_loss.item() / total_batch,
                      "Loss/cls_l1": cls_l1_loss.item() / total_batch,
                      "Loss/cls_location": cls_location_loss.item() / total_batch,
                      "Loss/scroe_ve": score_ve_l1_loss.item() / total_batch,
                      "IoU": mean_iou.item() / total_batch,}
            return loss, status
        else:
            return loss
        
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

def draw_tmbbox(imgs_,bboxs):
    imgs = imgs_.clone()
    mean = torch.tensor((0.4379, 0.4251, 0.4685)).reshape(1,1,1,3).to(imgs)
    std = torch.tensor((0.2361, 0.2416, 0.2415)).reshape(1,1,1,3).to(imgs)
    
    imgs = ((imgs.permute(0,2,3,1) *std + mean) * 255).to(torch.int32).contiguous()
    bboxs = bboxs.view(-1,4) * imgs.shape[1]
    id = 1
    for img, bbox in zip(imgs,bboxs):
        img = img.detach().cpu().numpy()
        bbox = bbox.to(torch.int32).tolist()
        img = cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,0,255),1)
        cv2.imwrite('MemTrack/visual_tm_bbox/'+str(id)+'.jpg',img)

def draw_shbbox(imgs_,bboxs):
    imgs = imgs_.clone()
    imgs = rearrange(imgs,'b c h w -> b h w c')
    mean = torch.tensor((0.485, 0.456, 0.406)).reshape(1,1,1,3).to(imgs)
    std = torch.tensor((0.229, 0.224, 0.225)).reshape(1,1,1,3).to(imgs)
    imgs = ((imgs *std + mean) * 255).to(torch.int32).contiguous()
    bboxs = bboxs * imgs.shape[1]
    bboxs[:,2] = bboxs[:,0] + bboxs[:,2]
    bboxs[:,3] = bboxs[:,1] + bboxs[:,3]
    id = 1
    for img, bbox in zip(imgs,bboxs):
        img = img.detach().cpu().numpy()
        bbox = bbox.to(torch.int32).tolist()
        img = cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),1)
        cv2.imwrite('MemTrack/visual_gt_bbox/'+str(id)+'.jpg',img)

def draw_hisbbox(imgs_,bboxs):
    imgs = imgs_.clone()
    bh = imgs.shape[1]
    for bs in range(bh):
        img = imgs[:,bs]
        bbox = bboxs[:,bs]
        img = rearrange(img,'b c h w -> b h w c')
        mean = torch.tensor((0.485, 0.456, 0.406)).reshape(1,1,1,3).to(img)
        std = torch.tensor((0.229, 0.224, 0.225)).reshape(1,1,1,3).to(img)
        img = ((img *std + mean) * 255).to(torch.int32).contiguous()
        img = img.detach().cpu().numpy()
        bbox = bbox * img.shape[1]
        id = 1
        for img_ ,boxs in zip(img,bbox):
            for bbox in boxs:
                bbox = bbox.to(torch.int32).tolist()
                img_ = cv2.rectangle(img_,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),1)
                cv2.imwrite('MemTrack/visual_his_bbox/'+str(id)+'.jpg',img_) 
    