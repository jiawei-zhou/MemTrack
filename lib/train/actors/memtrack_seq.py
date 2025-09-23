from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy,box_cxcywh_to_xywh
from lib.train.data.processing_utils import sample_target
import torch
import math
import numpy as np
import numpy
import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as tvisf
import lib.train.data.bounding_box_utils as bbutils
from lib.utils.merge import merge_template_search
from torch.distributions.categorical import Categorical
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from torchvision import utils
from einops import rearrange
from lib.test.utils.hann import hann2d

def IoU(rect1, rect2):
    """ caculate interection over union
    Args:
        rect1: (x1, y1, x2, y2)
        rect2: (x1, y1, x2, y2)
    Returns:
        iou
    """
    # overlap
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2 - x1) * (y2 - y1)
    target_a = (tx2 - tx1) * (ty2 - ty1)
    inter = ww * hh
    iou = inter / (area + target_a - inter)
    return iou


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



def SIoU_loss(test1, test2, theta=4):
    eps = 1e-7  # angle cost
    cx_pred = (test1[:, 0] + test1[:, 2]) / 2
    cy_pred = (test1[:, 1] + test1[:, 3]) / 2
    cx_gt = (test2[:, 0] + test2[:, 2]) / 2
    cy_gt = (test2[:, 1] + test2[:, 3]) / 2

    dist = ((cx_pred - cx_gt) ** 2 + (cy_pred - cy_gt) ** 2) ** 0.5
    ch = torch.max(cy_gt, cy_pred) - torch.min(cy_gt, cy_pred)
    x = ch / (dist + eps) # h / d

    angle = 1 - 2 * torch.sin(torch.arcsin(x) - torch.pi / 4) ** 2 # x与d的夹角 相较于45度的偏移
    # distance cost
    xmin = torch.min(test1[:, 0], test2[:, 0])
    xmax = torch.max(test1[:, 2], test2[:, 2])
    ymin = torch.min(test1[:, 1], test2[:, 1])
    ymax = torch.max(test1[:, 3], test2[:, 3])
    cw = xmax - xmin
    ch = ymax - ymin
    px = ((cx_gt - cx_pred) / (cw + eps)) ** 2
    py = ((cy_gt - cy_pred) / (ch + eps)) ** 2
    gama = 2 - angle
    dis = (1 - torch.exp(-1 * gama * px)) + (1 - torch.exp(-1 * gama * py))

    # shape cost
    w_pred = test1[:, 2] - test1[:, 0]
    h_pred = test1[:, 3] - test1[:, 1]
    w_gt = test2[:, 2] - test2[:, 0]
    h_gt = test2[:, 3] - test2[:, 1]
    ww = torch.abs(w_pred - w_gt) / (torch.max(w_pred, w_gt) + eps)
    wh = torch.abs(h_gt - h_pred) / (torch.max(h_gt, h_pred) + eps)
    omega = (1 - torch.exp(-1 * wh)) ** theta + (1 - torch.exp(-1 * ww)) ** theta

    # IoU loss
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

    c2 = cw ** 2 + ch ** 2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
    rho2 = left + right

    factor = 4 / math.pi ** 2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    # CIoU
    cious = ious - (rho2 / c2 + v ** 2 / (1 - ious + v))
    return cious, ious


class MemTrackSeqActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, bins, search_size, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.bins = bins
        self.search_size = search_size
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.smooth_l1 = torch.nn.SmoothL1Loss(beta=0.1,reduction='mean')
        self.focal = None
        self.cls_loss = None
        self.range = cfg.MODEL.RANGE
        self.pre_num = cfg.MODEL.SEQNUMS
        self.pre_bbox = None
        self.x_feat_rem = None
        self.update_rem = None
        self.output_sz = {}
        self.output_sz['template'] = self.cfg.DATA.TEMPLATE.SIZE
        self.output_sz['search'] = self.cfg.DATA.SEARCH.SIZE
        self.output_window = hann2d(torch.tensor([cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE, cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE]).long(), centered=True).cuda()

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

        # compute losses
        loss, status = self.compute_sequence_losses(out_dict, data)

        return loss, status

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height
    
    def target_mask(self,bbox,mode='search',downsample=True):
        bboxs = bbox
        mask_list = []
        gauss_mask_lsit = []
        pure_gauss_mask_list = []
        for bbox in bboxs:
            mask = torch.zeros(self.output_sz[mode],self.output_sz[mode])
            bbox = bbox.to(torch.int32).tolist()
            x1,y1 = bbox[0],bbox[1]
            x2,y2 = bbox[2],bbox[3]
            mask[y1:y2,x1:x2] = 1   # utils.save_image(out,'mask.jpg')
            mask_list.append(mask.reshape(1,*mask.shape).to(torch.float32))
            gauss_mask,pure_gauss_mask = self.create_gaussian_mask(mask)
            gauss_mask_lsit.append(gauss_mask.reshape(1,*mask.shape).to(torch.float32))
            pure_gauss_mask_list.append(pure_gauss_mask.reshape(1,*mask.shape).to(torch.float32))
        mask = torch.stack(mask_list)
        gauss_mask = torch.stack(gauss_mask_lsit)
        pure_gauss_mask = torch.stack(pure_gauss_mask_list)
        if downsample:
            hw = self.output_sz[mode] // self.cfg.MODEL.BACKBONE.STRIDE
            mask = F.interpolate(mask,(hw,hw))
            gauss_mask = F.interpolate(gauss_mask,(hw,hw))
            pure_gauss_mask = F.interpolate(pure_gauss_mask,(hw,hw),mode='bilinear')
        return mask,gauss_mask,pure_gauss_mask
    
    def create_gaussian_mask(self,mask,return_parameters=False):
        H, W = mask.shape
        center_y, center_x = torch.nonzero(mask).float().mean(dim=0).long()
        h_idx,w_idx = torch.where(mask>0)
        h,w = len(torch.unique(h_idx)),len(torch.unique(w_idx))
        h,w = torch.tensor((h)).to(mask),torch.tensor((w)).to(mask)
        min_hw,max_hw = min(h,w),max(h,w)
        hw = torch.stack((min_hw,max_hw*0.9),dim=0) / 2
        sigma = torch.sqrt(torch.square(hw).sum())
        # Create a grid of (x, y) coordinates
        y = torch.arange(H).float().unsqueeze(1).expand(H, W)
        x = torch.arange(W).float().unsqueeze(0).expand(H, W)

        # Calculate the distance from each point to the center
        distance = ((x - center_x.float())**2 + (y - center_y.float())**2).sqrt()
        
        # Generate Gaussian distribution without mask
        gaussian = torch.exp(-0.5 * (distance / sigma)**2)
        
        # Apply the mask to keep only the values where mask is 1
        gaussian_masked = gaussian * mask.float()

        sigma = min_hw * 0.7
        # Calculate the distance from each point to the center
        distance = ((x - center_x.float())**2 + (y - center_y.float())**2).sqrt()
        
        # Generate Gaussian distribution
        gaussian = torch.exp(-0.5 * (distance / sigma)**2)
        
        if not return_parameters:
            return gaussian_masked,gaussian
        return gaussian_masked,center_y, center_x,sigma
    
    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            if k != 3:
                te_im = np.zeros(size,np.float32)
            else:
                te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            try:
                im_patch = cv2.resize(im_patch, (model_sz, model_sz))
            except:
                return None
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        im_patch = im_patch.cuda()
        return im_patch

    def batch_init(self, images,template_bbox, initial_bbox) -> dict:
        self.frame_num = 0
        self.device = 'cuda'
        # Convert bbox (x1, y1, w, h) -> (cx, cy, w, h)
        template_bbox = template_bbox.reshape(-1,4)
        initial_bbox = initial_bbox.reshape(-1,4)
        # template_bbox = bbutils.batch_xywh2center2(template_bbox)  # ndarray:(2*num_seq,4)
        initial_bbox = bbutils.batch_xywh2center2(initial_bbox)  # ndarray:(2*num_seq,4)
        self.center_pos = initial_bbox[:, :2]  # ndarray:(2*num_seq,2)
        self.size = initial_bbox[:, 2:]  # ndarray:(2*num_seq,2)
    
        template_factor = self.cfg.DATA.TEMPLATE.FACTOR
        search_factor = self.cfg.DATA.SEARCH.FACTOR
        w_x = initial_bbox[:, 2] * search_factor  # ndarray:(2*num_seq)
        h_x = initial_bbox[:, 3] * search_factor  # ndarray:(2*num_seq)
        s_x = np.ceil(np.sqrt(w_x * h_x))  # ndarray:(2*num_seq)

        # self.channel_average = []
        # for img in images:
        #     self.channel_average.append(np.mean(img, axis=(0, 1)))
        # self.channel_average = np.array(self.channel_average)  # ndarray:(2*num_seq,3)
        
        # get crop
        z_crop_list = []
        temp_region_label_list =[]
        temp_region_bbox_xywh_list = []
        for i in range(len(images)):
            im_crop_padded, resize_factor, att_mask = sample_target(images[i][0], template_bbox[i],template_factor,
                                                                                      output_sz=self.cfg.DATA.TEMPLATE.SIZE,)
            here_crop = rearrange(torch.from_numpy(im_crop_padded).unsqueeze(0),'b h w c -> b c h w').to(torch.float32)
            z_crop = here_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            self.inplace = False
            z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)
            z_crop_list.append(z_crop.clone())
            
            temp_region_wh = torch.from_numpy(initial_bbox[[i], 2:] * resize_factor)
            temp_region_cxcy = torch.tensor((self.cfg.DATA.TEMPLATE.SIZE//2,self.cfg.DATA.TEMPLATE.SIZE//2)).reshape(-1,2)
            temp_region_bbox = torch.cat([temp_region_cxcy,temp_region_wh],dim=-1)
            temp_region_bbox_xyxy = box_cxcywh_to_xyxy(temp_region_bbox)
            temp_region_bbox_xywh = box_cxcywh_to_xywh(temp_region_bbox)
            temp_region_label,_,_ = self.target_mask(bbox=temp_region_bbox_xyxy,mode='template')
            temp_region_label_list.append(temp_region_label)
            temp_region_bbox_xywh_list.append(temp_region_bbox_xywh)
        
        z_crop = torch.cat(z_crop_list, dim=0).to(self.device)  # Tensor(2*num_seq,3,128,128)
        temp_region_label = torch.cat(temp_region_label_list,dim=0).to(z_crop)
        temp_bbox = torch.stack(temp_region_bbox_xywh_list,dim=0)
        box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, z_crop.shape[0], z_crop.device,
                                            temp_bbox)

        search_region_wh = torch.from_numpy(initial_bbox[:, 2:] * self.cfg.DATA.SEARCH.SIZE / s_x.reshape(-1,1))
        self.search_wh = search_region_wh.unsqueeze(1)
        self.prev_area = torch.prod(search_region_wh,dim=-1).reshape(-1,1)
        self.score = torch.zeros(z_crop.shape[0],self.cfg.MODEL.MEMORYBLOCK.MEMORY_MAX_NUM).to(z_crop.device)
        self.score[:,0] = 1
        # hsi_z_crop = torch.cat(hsi_z_crop_list,dim=0)
        self.update_rem = None
        # utils.save_image(here_hsi_crop[:,:3,:,:],'t.jpg')
        out = {'template_images': z_crop,
               'template_label':temp_region_label,
               'box_mask_z':box_mask_z}
        return out

    def batch_track(self, img, gt_boxes, template,template_label,box_mask_z) -> dict:
        self.frame_num += 1
        search_factor = self.cfg.DATA.SEARCH.FACTOR
        crop_bbox = bbutils.batch_cxcywh2xywh(np.concatenate([self.center_pos,self.size],axis=-1)) # cx cy w h -> x y w h
        gt_boxes_corner = bbutils.batch_xywh2corner(gt_boxes)  # ndarray:(2*num_seq,4)
        x_crop_list = []
        gt_in_crop_list = []
        gt_mask_list = []
        gt_gauss_mask_list = []
        score_ve_label_list = []
        resize_factor_list = []

        # utils.save_image(x_crop.clone().float().mul(1.0 / 255.0).clamp(0.0, 1.0),'simg.jpg')
        # cv2.imwrite('simg.jpg',img[i])

        for i in range(len(img)):   # 从batch个搜索图像中依次取出1个搜索图像去处理。 这一段代码描述的是利用目标历史信息
            im_crop_padded, resize_factor, att_mask = sample_target(img[i], crop_bbox[i],search_factor,
                                                                    output_sz=self.cfg.DATA.SEARCH.SIZE,)
            resize_factor_list.append(resize_factor)
            if (im_crop_padded == None).all():
                return None

            if gt_boxes_corner is not None and np.sum(np.abs(gt_boxes_corner[i] - np.zeros(4))) > 10:
                gt_in_crop = np.zeros(4)
                gt_in_crop[:2] = gt_boxes_corner[i, :2] - self.center_pos[i]
                gt_in_crop[2:] = gt_boxes_corner[i, 2:] - self.center_pos[i]
                gt_in_crop = gt_in_crop * resize_factor + self.cfg.DATA.SEARCH.SIZE / 2
                gt_mask,gt_gauss_mask,score_ve_label = self.target_mask(torch.from_numpy(gt_in_crop).reshape(-1,4))
                gt_mask_list.append(gt_mask)
                gt_gauss_mask_list.append(gt_gauss_mask)
                score_ve_label_list.append(score_ve_label)
                gt_in_crop[2:] = gt_in_crop[2:] - gt_in_crop[:2]  # (x1,y1,x2,y2) to (x1,y1,w,h)
                gt_in_crop_list.append(gt_in_crop / self.cfg.DATA.SEARCH.SIZE)
            else:
                gt_in_crop_list.append(np.zeros(4))

            x_crop = rearrange(torch.from_numpy(im_crop_padded).unsqueeze(0),'b h w c -> b c h w').to(torch.float32)
            x_crop = x_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
            x_crop[0] = tvisf.normalize(x_crop[0], self.mean, self.std, self.inplace)
            x_crop_list.append(x_crop.clone())

        # utils.save_image(hsi_x_crop_list[5][:,:3],'simg.jpg')
        #self.draw_bbox(x_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0),torch.tensor(gt_in_crop))
        x_crop = torch.cat(x_crop_list, dim=0).to(self.device)
        gt_mask = torch.cat(gt_mask_list,dim=0)
        gt_gauss_mask = torch.cat(gt_gauss_mask_list,dim=0)
        score_ve_label = torch.cat(score_ve_label_list,dim=0)
 
        # self.draw_bbox(x_crop,torch.tensor(np.stack(gt_in_crop_list, axis=0), dtype=torch.float))

        # search_feature is useless frame_id,template,search,target_label,ce_template_mask,ce_keep_rate
        outputs = self.net(mode='track',frame_id=self.frame_num,template=template,search=x_crop,template_pixle_label=template_label,ce_template_mask=box_mask_z,ce_keep_rate=None)
        pred_boxes = outputs['bbox']
        pred_score_map = outputs['score_map']
        pred_label = outputs['pred_label']
        backbone_feat = outputs['backbone_feat']
        score_ve = outputs['score_ve']
        memory_list = []
        update_list = []
        replace_index_list = []
        # update memory 
        for i in range(len(pred_label)):
            update,_ = self.determine_updatemMemory(i,pred_label[i:i+1])
            if update and self.frame_num < self.score.shape[1]:
                update_list.append(torch.ones(1))
                replace_index_list.append(None)
                self.score[i,self.frame_num] = score_ve[i].max()
                memory = self.net(mode='Memencode',feat=backbone_feat[i:i+1],mask=pred_label[i:i+1])
            elif update and (score_ve[i].max()>=self.score[i]).any():
                update_list.append(torch.ones(1))
                index = torch.where(score_ve[i].max()>=self.score[i])[0][0]
                replace_index_list.append(index)
                self.score[i,index] = score_ve[i].max()
                memory = self.net(mode='Memencode',feat=backbone_feat[i:i+1],mask=pred_label[i:i+1])
            else:
                update_list.append(torch.zeros(1))
                memory = self.net(mode='getMemory')[0,i:i+1]
                replace_index_list.append(None)
            memory_list.append(memory)

        update_index = torch.cat(update_list)
        memory = torch.cat(memory_list,dim=0)
        self.net(mode='addMemory',score_ve=score_ve,in_memorys=memory,update_index=update_index,replace_index=replace_index_list) 

        # add hann windows
        response = self.output_window * pred_score_map
        pred_boxes = self.net(mode='getbbox',response=response,size_map=outputs['size_map'], offset_map=outputs['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        search_bbox = (pred_boxes * self.cfg.DATA.SEARCH.SIZE).detach().cpu()
        self.search_wh = torch.cat([self.search_wh,search_bbox[:,2:].unsqueeze(1)],dim=1)
        search_area = torch.prod(search_bbox[:,2:],dim=-1).reshape(-1,1)
        self.prev_area = torch.cat([self.prev_area,search_area],dim=-1)

        cx=cy=w=h= None
        # get the final box result
        for i in range(len(img)):
            pred_box = (pred_boxes[i] * self.cfg.DATA.SEARCH.SIZE / resize_factor_list[i]) # (cx, cy, w, h) [0,1]
            pred_box = pred_box.detach().cpu().numpy()
            s_x = self.cfg.DATA.SEARCH.SIZE / resize_factor_list[i]
            cx_ = pred_box[0] + self.center_pos[i,0] - s_x / 2  
            cy_ = pred_box[1] + self.center_pos[i,1] - s_x / 2
            w_, h_ = pred_box[2], pred_box[3]
            cx_,cy_,w_,h_ = self._bbox_clip(cx_, cy_, w_,h_, img[i].shape[:2])
            cx_,cy_,w_,h_ = np.array(cx_).reshape(1),np.array(cy_).reshape(1),np.array(w_).reshape(1),np.array(h_).reshape(1)
            if cx is None:
                cx,cy,w,h = cx_,cy_,w_,h_
            else:
                cx = np.concatenate([cx,cx_])
                cy = np.concatenate([cy,cy_])
                w = np.concatenate([w,w_])
                h = np.concatenate([h,h_])

        self.center_pos = np.stack([cx, cy], 1)
        self.size = np.stack([w, h], 1)
        bbox = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
        
        # x_feat = outputs['x_feat'].detach().cpu()
        # self.x_feat_rem = x_feat.clone()
        # x_feat_list.append(x_feat.clone())

        out = {
            'search_images': x_crop.detach().cpu(),
            'pred_bboxes': bbox,
            'update_index': update_index.detach().cpu(),
            'gt_in_crop': torch.tensor(np.stack(gt_in_crop_list, axis=0), dtype=torch.float).detach().cpu(),
            'pred_label': pred_label.detach().cpu(),
            'gt_mask':gt_mask,
            'gt_gauss_mask':gt_gauss_mask,
            'score_ve_label':score_ve_label,
            'replace_index':replace_index_list,
            # 'x_feat': torch.tensor([item.cpu().detach().numpy() for item in x_feat_list], dtype=torch.float),
        }
        return out
    
    def determine_updatemMemory(self,bs,pred_label):
        prev_w,prev_h = self.search_wh[bs,-1,0],self.search_wh[bs,-1,1]
        prev_area = self.prev_area[bs].mean()
        label = F.interpolate(pred_label.to(torch.float32),self.cfg.DATA.SEARCH.SIZE).cpu()
        pred_label_area = label.sum()
        pred_cx,pred_cy = (torch.where(label==1)[2].unique().to(torch.float32)+1e-4).mean(),(torch.where(label==1)[3].unique().to(torch.float32)+ 1e-4).mean()
        prev_cx = prev_cy = self.cfg.DATA.SEARCH.SIZE /2
        if prev_w <= prev_h:
            ratio = torch.tensor((prev_w/ prev_h))
            threshold_h = torch.exp(ratio-1) * self.search_wh[bs,:,1].mean() 
            isin_w = abs(pred_cx - prev_cx) < self.search_wh[bs,:,0].mean() 
            isin_h = abs(pred_cy - prev_cy) < threshold_h
        else:
            ratio = torch.tensor((prev_h / prev_w))
            threshold_w = torch.exp(ratio-1) * self.search_wh[bs,:,0].mean() 
            isin_w = abs(pred_cx - prev_cx)  < threshold_w
            isin_h = abs(pred_cy - prev_cy)  < self.search_wh[bs,:,1].mean() 
        # isin_w = abs(pred_cx - prev_cx) / resize_factor < prev_w
        # isin_h = abs(pred_cy - prev_cy) / resize_factor < prev_h
        # distance = torch.sqrt((pred_cx-prev_cx)**2 + (pred_cy-prev_cy)**2) 
        # threshold_dist = math.sqrt(min(prev_h,prev_w) ** 2 + (max(prev_h,prev_w)/2) ** 2)
        if (pred_label_area > 0.7 *prev_area and pred_label_area < 1.2*prev_area) and (isin_w and isin_h):
            return True,True
        elif (isin_w and isin_h):
            return False,True
        else:
            return False,False
        
    def draw_bbox(self,imgs,bboxs):
        """
        bboxs: x,y,w,h
        """
        if not isinstance(imgs,list):
            mean = torch.tensor((0.485, 0.456, 0.406)).reshape(1,1,1,3).to(imgs)
            std = torch.tensor((0.229, 0.224, 0.225)).reshape(1,1,1,3).to(imgs)
            imgs = ((imgs.permute(0,2,3,1) *std + mean) * 255).to(torch.int32).contiguous()
        bboxs = bboxs.view(-1,4)
        id = 0
        for img, bbox in zip(imgs,bboxs):
            id = 1
            if isinstance(imgs,list):
                img = img.permute(0,2,3,1).reshape(*img.shape[2:],-1)
                img = img[:,:,:3] * 255 
            img = img.detach().cpu().numpy().copy().astype(np.int32)
            bbox = bbox.to(torch.int32).tolist()
            img = cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,0,255),1)
            cv2.imwrite(str(id)+'.jpg',img)

    def explore(self, data):
        results = {}
        search_images_list = []
        search_anno_list = []
        iou_list = []
        pred_label_list = []
        gt_mask_list = []
        gt_gauss_mask_list =[]
        score_ve_label_list = []
        update_index_list =[]
        replace_index_list = []

        num_frames = data['search_images'][0].shape[0]
        images = data['search_images']
        gt_bbox = data['search_anno']
        template = data['template_images']
        template_bbox = data['template_anno']
        num_seq = len(template)
        template_bbox = np.array(template_bbox).reshape(-1,4)
        
        # 循环取出所有图片对，一次取出batch对
        for idx in range(np.max(num_frames)):
            here_images = [img[idx] for img in images]  # S, N
            here_gt_bbox = np.array([gt[idx] for gt in gt_bbox])
            here_gt_bbox = np.concatenate([here_gt_bbox], 0)
            
            if idx == 0:
                outputs_template = self.batch_init(template,template_bbox, here_gt_bbox)
                results['template_images'] = outputs_template['template_images'].clone().detach().cpu()
                results['template_label'] = outputs_template['template_label'].clone().detach().cpu()
                results['box_mask_z'] = outputs_template['box_mask_z'].clone().detach().cpu()
            else:
                outputs = self.batch_track(here_images,here_gt_bbox,outputs_template['template_images'],
                                           outputs_template['template_label'],outputs_template['box_mask_z'])
                if outputs == None:
                    return None
                
                # x_feat = outputs['x_feat']
                pred_bbox = outputs['pred_bboxes']
                pred_label_list.append(outputs['pred_label'])
                gt_mask_list.append(outputs['gt_mask'])
                gt_gauss_mask_list.append(outputs['gt_gauss_mask'])
                score_ve_label_list.append(outputs['score_ve_label'])
                update_index_list.append(outputs['update_index'])
                search_images_list.append(outputs['search_images'])
                search_anno_list.append(outputs['gt_in_crop'])
                replace_index_list.append(outputs['replace_index'])
                # if len(outputs['pre_seq']) != 8:
                #     print(outputs['pre_seq'])
                #     print(len(outputs['pre_seq']))
                #     print(idx)
                #     print(data['num_frames'])
                #     print(data['search_annos'])
                #     return None

                pred_bbox_corner = bbutils.batch_xywh2corner(pred_bbox)
                gt_bbox_corner = bbutils.batch_xywh2corner(here_gt_bbox)
                here_iou = []
                for i in range(num_seq):
                    bbox_iou = IoU(pred_bbox_corner[i], gt_bbox_corner[i])
                    here_iou.append(bbox_iou)
                iou_list.append(here_iou)
                # x_feat_list.append(x_feat.clone())

        # results['x_feat'] = torch.cat([torch.stack(x_feat_list)], dim=2)
        try:
            results['search_images'] = torch.stack(search_images_list,dim=1)
            results['search_anno'] = torch.stack(search_anno_list,dim=1)
            results['pred_label'] = torch.stack(pred_label_list, dim=1) 
            results['gt_mask'] = torch.stack(gt_mask_list,dim=1) 
            results['gt_gauss_mask'] = torch.stack(gt_gauss_mask_list,dim=1)  
            results['score_ve_label'] = torch.stack(score_ve_label_list, dim=1) 
            results['update_index'] = torch.stack(update_index_list, dim=1)  
            iou_tensor = torch.tensor(iou_list, dtype=torch.float)
            results['baseline_iou'] = torch.cat([iou_tensor[:, :num_seq]], dim=1)
            results['replace_index'] = replace_index_list
        except Exception as e:
            print(e)
            return None
        return results
    
    def compute_sequence_losses(self, data,return_status=True):
        num_frames = data['search_images'].shape[0]
        template_images = data['template_images'].repeat(num_frames, 1, 1, 1)
        template_label = data['template_label']
        box_mask_z = data['box_mask_z'].repeat(num_frames,1)
        search_images = data['search_images']
        search_anno = data['search_anno']
        search_label = data['pred_label']
        gt_mask = data['gt_mask']
        gt_gauss_mask = data['gt_gauss_mask']
        score_ve_label = data['score_ve_label']
        update_index = data['update_index']
        replace_index = data['replace_index']
        
        outputs = self.net(template=template_images,search=search_images,ce_template_mask=box_mask_z,ce_keep_rate=None,
                           template_label=template_label,template_pixle_label=template_label,search_label=search_label,
                           update_index=update_index,replace_index=replace_index)

        gt_bbox = search_anno # (Ns, batch, 4) (x1,y1,w,h) -> (Ns * batch, 4)
        gt_gaussian_maps = generate_heatmap(search_anno.unsqueeze(0), self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = torch.stack(gt_gaussian_maps,dim=1)
        # B,C,H,W = gt_gaussian_maps.shape
        # pad = torch.ones(B,C,H,1).to(gt_gaussian_maps)
        # utils.save_image(torch.cat([gt_gaussian_maps,pad],dim=-1),'gauss.jpg')
        # Get boxes
        pred_boxes = outputs['bbox']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (n,b,4) --> (n*b,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0,max=1.0) # (Ns * batch, 4) (x1,y1,x2,y2)
        # gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
        #                                                                                                    max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            head_giou_loss, head_iou = SIoU_loss(pred_boxes_vec, gt_boxes_vec,4)  # (BN,4) (BN,4)
            head_giou_loss = head_giou_loss.mean()
        except:
            head_giou_loss, head_iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        head_l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        # compute contrast loss
        #contrastive_loss = self.objective['contrast'](pred_dict['pred_contrast'])
        # compute location loss
        if 'score_map' in outputs:
            score_map = outputs['score_map']
            head_location_loss = self.objective['focal'](score_map, gt_gaussian_maps)
        else:
            head_location_loss = torch.tensor(0.0, device=head_l1_loss.device)
        # compute clssification loss
        if self.cls_loss is None:
            cls_weigth = torch.ones(2).to(gt_boxes_vec)
            self.cls_loss = torch.nn.CrossEntropyLoss(weight=cls_weigth, size_average=True).to(gt_boxes_vec)

        pred_label_feat = rearrange(outputs['pred_label_feat'],'b c h w -> (b h w) c')
        gt_lable_ce = rearrange(gt_mask,'b c h w -> (b h w c)').to(torch.int64)
        cls_location_loss = self.cls_loss(pred_label_feat,gt_lable_ce)

        gt_lable_l1 = gt_gauss_mask
        pred_label = outputs['pred_label_feat'].softmax(dim=1)[:,[1]]
        cls_l1_loss = self.objective['l1'](pred_label,gt_lable_l1)
        
        # # # compute score_ve loss
        # score_ve = outputs['score_ve']
        # score_ve = (score_ve - score_ve.min(dim=-1,keepdim=True)[0]) / (score_ve.max(dim=-1,keepdim=True)[0] - score_ve.min(dim=-1,keepdim=True)[0])
        # # scroe_ve_location_loss = self.objective['focal'](scroe_ve, gt_gaussian_maps)
        # score_ve_l1_loss = self.smooth_l1(score_ve,rearrange(score_ve_label,'b c h w -> b (c h w)'))
        loss = self.loss_weight['giou'] * (head_giou_loss) + self.loss_weight['l1'] * (head_l1_loss + cls_l1_loss) + \
                self.loss_weight['focal'] * (head_location_loss + cls_location_loss) 
        
        if return_status:
            # status for log
            mean_iou = head_iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": head_giou_loss.item(),
                      "Loss/head_location": head_location_loss.item(),
                      "Loss/head_l1": head_l1_loss.item(),
                      "Loss/cls_l1": cls_l1_loss.item(),
                      "Loss/cls_location": cls_location_loss.item(),
                    #   "Loss/scroe_ve": score_ve_l1_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
    # mobileSAM 速度太慢了
    # def create_mask(self,imgs,bbox,his_bbox=None,vis=False):
    #     """## bbox reduce the first 

    #     ### Args:
    #         - `imgs (tenosr)`: n,b,c,h,w
    #         - `bbox (tenosr)`: n,b,4
    #     """
    #     # transfer numpy and reversed normalize
    #     N,B,C,H,W = imgs.shape
    #     imgs = rearrange(imgs,'n b c h w -> n b h w c').clone().detach().cpu()
    #     mean = self.mean.reshape(1,1,1,1,3).to(imgs)
    #     std = self.std.reshape(1,1,1,1,3).to(imgs)
    #     imgs = ((imgs *std + mean) * 255).numpy().astype(np.uint8)
    #     # imgs to batch list
    #     # imgs = [i for i in imgs]
    #     # bbox to batch list and remove normalize
    #     bbox = rearrange(bbox.clone().detach().cpu().numpy() * H,'(n b) c -> n b c',b=self.bs)
    #     bbox[:,:,2:] = bbox[:,:,2:] + bbox[:,:,:2]
    #     bboxs = bbox.astype(np.int32)
    #     if his_bbox is not None:
    #         his_bbox = his_bbox.clone().detach().cpu().numpy() * H
    #         his_bbox = his_bbox.astype(np.int32)
    #     # bbox = [i for i in bbox]
    #     masks_batch = []
    #     # single image predictor
    #     with torch.inference_mode(),torch.autocast("cuda", dtype=torch.bfloat16):
    #         imgs = imgs.reshape(-1,H,W,C)
    #         bboxs = bboxs.reshape(-1,4)
    #         if his_bbox is not None:
    #             his_bbox = his_bbox.reshape(N*B,-1,4)
    #         for i in range(N*B):
    #             img = imgs[i]
    #             bbox = bboxs[i]
    #             if his_bbox is not None:
    #                 his_box = his_bbox[i]
    #             self.sam.set_image(img)
    #             masks, scores, _ = self.sam.predict(point_coords=None,
    #                                                 point_labels=None,
    #                                                 box=bbox[None, :],
    #                                                 multimask_output=False,)
    #             if vis:
    #                 cv2.imwrite('img.jpg',img)
    #                 plt.figure(figsize=(10, 10))
    #                 plt.imshow(img)
    #                 show_mask(masks[0], plt.gca())
    #                 show_box(bbox, plt.gca())
    #                 plt.axis('off')
    #                 plt.savefig('mask.jpg')
    #             masks_batch.append(masks)
    #             torch.cuda.empty_cache()
    
           
    def create_mask(self,imgs,bbox,his_bbox=None,vis=False):
        """## bbox reduce the first 

        ### Args:
            - `imgs (tenosr)`: n,b,c,h,w
            - `bbox (tenosr)`: n,b,4
        """
        # transfer numpy and reversed normalize
        N,B,C,H,W = imgs.shape
        imgs = rearrange(imgs,'n b c h w -> n b h w c').clone().detach().cpu()
        mean = self.mean.reshape(1,1,1,1,3).to(imgs)
        std = self.std.reshape(1,1,1,1,3).to(imgs)
        imgs = ((imgs *std + mean) * 255).numpy().astype(np.uint8)
        # imgs to batch list
        # imgs = [i for i in imgs]
        # bbox to batch list and remove normalize
        bbox = rearrange(bbox.clone().detach().cpu().numpy() * H,'(n b) c -> n b c',b=self.bs)
        bbox[:,:,2:] = bbox[:,:,2:] + bbox[:,:,:2]
        bboxs = bbox.astype(np.int32)
        if his_bbox is not None:
            his_bbox = his_bbox.clone().detach().cpu().numpy() * H
            his_bbox = his_bbox.astype(np.int32)
        # bbox = [i for i in bbox]
        masks_batch = []
        # # single image predictor
        # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        #     imgs = imgs.reshape(-1,H,W,C)
        #     bboxs = bboxs.reshape(-1,4)
        #     his_bbox = his_bbox.reshape(N*B,-1,4)
        #     for i in range(N*B):
        #         img = imgs[i]
        #         bbox = bboxs[i]
        #         if his_bbox is not None:
        #             his_box = his_bbox[i]
        #         self.sam.set_image(img)
        #         masks, scores, _ = self.sam.predict(point_coords=None,
        #                                             point_labels=None,
        #                                             box=bbox[None, :],
        #                                             multimask_output=False,)
        #         if vis:
        #             cv2.imwrite('img.jpg',img)
        #             show_masks(img, masks, scores, box_coords=bbox,his_bbox=his_box)
        #         masks_batch.append(masks)
        #         torch.cuda.empty_cache()
        # all batch images predictor
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            imgs = imgs.reshape(-1,H,W,C)
            bboxs = bboxs.reshape(-1,4)
            img = [i for i in imgs]
            bbox = [i for i in bboxs]
            if his_bbox is not None:
                his_bbox = his_bbox.reshape(N*B,-1,4)
                his_box = his_bbox
            self.sam.set_image_batch(img)
            masks_, scores, _ = self.sam.predict_batch(None,None,box_batch=bbox,multimask_output=False)
            masks_batch.append(np.stack(masks_))
            if vis:
                if his_bbox is None:
                    for image, boxes, masks in zip(img, bbox, masks_):
                        plt.figure(figsize=(10, 10))
                        plt.imshow(image) 
                        cv2.imwrite('img.jpg',image)  
                        for mask in masks:
                            show_mask(mask, plt.gca(), random_color=True)
                        show_box(boxes, plt.gca())
                        if his_bbox is not None:
                            for box in his_bbox:
                                show_box(box, plt.gca())
                        plt.savefig('mask.jpg')
                else:
                    for image, boxes, masks,his_boxes in zip(img, bbox, masks_,his_box):
                        plt.figure(figsize=(10, 10))
                        plt.imshow(image) 
                        cv2.imwrite('img.jpg',image)  
                        for mask in masks:
                            show_mask(mask, plt.gca(), random_color=True)
                        show_box(boxes, plt.gca())
                        for box in his_boxes:
                            show_box(box, plt.gca())
                        plt.savefig('mask.jpg')
            torch.cuda.empty_cache()

        # batch images predictor
        # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        #     for i in range(B):
        #         img = [i for i in imgs[:,i]]
        #         bbox = [i for i in bboxs[:,i]]
        #         if his_bbox is not None:
        #             his_box = his_bbox[:,i]
        #         self.sam.set_image_batch(img)
        #         masks_, scores, _ = self.sam.predict_batch(None,None,box_batch=bbox,multimask_output=False)
        #         masks_batch.append(np.stack(masks_))
        #         if vis:
        #             if his_bbox is None:
        #                 for image, boxes, masks in zip(img, bbox, masks_):
        #                     plt.figure(figsize=(10, 10))
        #                     plt.imshow(image) 
        #                     cv2.imwrite('img.jpg',image)  
        #                     for mask in masks:
        #                         show_mask(mask, plt.gca(), random_color=True)
        #                     show_box(boxes, plt.gca())
        #                     if his_bbox is not None:
        #                         for box in his_bbox:
        #                             show_box(box, plt.gca())
        #                     plt.savefig('mask.jpg')
        #             else:
        #                 for image, boxes, masks,his_boxes in zip(img, bbox, masks_,his_box):
        #                     plt.figure(figsize=(10, 10))
        #                     plt.imshow(image) 
        #                     cv2.imwrite('img.jpg',image)  
        #                     for mask in masks:
        #                         show_mask(mask, plt.gca(), random_color=True)
        #                     show_box(boxes, plt.gca())
        #                     for box in his_boxes:
        #                         show_box(box, plt.gca())
        #                     plt.savefig('mask.jpg')
        #         torch.cuda.empty_cache()
            # for image, boxes, masks in zip(imgs, bbox, masks_batch):
            #     plt.figure(figsize=(10, 10))
            #     ax = plt.gca()
            #     ax.imshow(img)   
            #     cv2.imwrite('img.jpg',img)
            #     show_mask(mask, ax, random_color=True)
            #     show_box(bbox,ax)
            #     if vis_hisbox:
            #         for box in his_bbox[i]:
            #             show_box(box,ax)
            #     plt.savefig('mask.jpg')
                
        return masks_batch
        
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
        cv2.imwrite('/home/zhoujiawei/my_modelTrack/visual_maskimg/'+str(id)+'.jpg',img)

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
            
# def show_mask(mask, ax, random_color=False, borders = True):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask = mask.astype(np.uint8)
#     mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     if borders:
#         import cv2
#         contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
#         # Try to smooth contours
#         contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
#         mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
#     ax.imshow(mask_image)
    
# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=2))    

# def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True,his_bbox=None):
#     for i, (mask, score) in enumerate(zip(masks, scores)):
#         plt.figure(figsize=(10, 10))
#         plt.imshow(image)
#         show_mask(mask, plt.gca(), borders=borders)
#         if point_coords is not None:
#             assert input_labels is not None
#             show_points(point_coords, input_labels, plt.gca())
#         if box_coords is not None:
#             # boxes
#             show_box(box_coords, plt.gca())
#         if his_bbox is not None:
#             for box in his_bbox:
#                 show_box(box, plt.gca())
#         if len(scores) > 1:
#             plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#         plt.axis('off')
#         plt.savefig('mask.jpg')
#         plt.show()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 