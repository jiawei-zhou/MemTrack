import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F
import torchvision.transforms.functional as tvisf
import lib.train.data.bounding_box_utils as bbutils
import numpy as np
import cv2 as cv
from torchvision import utils
import math
from torchvision import transforms
import os
import cv2
import matplotlib.pyplot as plt
from einops import rearrange
import time
def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform}
        
    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None,sam=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.inplace = False
        self.settings = settings
        # self.sam = sam
        # self.mean = torch.tensor((0.485, 0.456, 0.406))
        # self.std = torch.tensor((0.229, 0.224, 0.225))

    def batch_clip_bbox_normal(self,bbox,im_size):
        x2,y2 = bbox[:,0] + bbox[:,2], bbox[:,1] + bbox[:,3]
        bbox[:,0] =  torch.clip(bbox[:,0],0,1-2/im_size)
        bbox[:,1] = torch.clip(bbox[:,1],0,1-2/im_size)
        x2 = torch.clip(x2,2/im_size,1-1/im_size)
        y2 = torch.clip(y2,2/im_size,1-1/im_size)
        w,h = x2 - bbox[:,0], y2-bbox[:,1]
        bbox[:,2],bbox[:,3] = w,h
        return bbox
    
    def clip_bbox_normal(self,bbox,im_size):
        x2,y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
        bbox[0] =  torch.clip(bbox[0],0,1-2/im_size)
        bbox[1] = torch.clip(bbox[1],0,1-2/im_size)
        x2 = torch.clip(x2,2/im_size,1-1/im_size)
        y2 = torch.clip(y2,2/im_size,1-1/im_size)
        w,h = x2 - bbox[0], y2-bbox[1]
        bbox[2],bbox[3] = w,h
        return bbox
    
    def clip_hisbbox_normal(self,bbox,im_size):
        bbox[:,0] =  torch.clip(bbox[:,0],0,1-2/im_size)
        bbox[:,1] = torch.clip(bbox[:,1],0,1-2/im_size)
        bbox[:,2] = torch.clip(bbox[:,2] ,2/im_size,1-1/im_size)
        bbox[:,3] = torch.clip(bbox[:,3],2/im_size,1-1/im_size)
        return bbox
    
    def clip_box(self,bbox, H, W,type='numpy'):

        x1, y1, x2, y2 = bbox[0],bbox[1],bbox[2],bbox[3]
        if type == 'torch':
            bbox[0] = torch.clip(x1,0,W-2)
            bbox[1] = torch.clip(y1,0,H-2)
            bbox[2] = torch.clip(x2,2,W-1)
            bbox[3] = torch.clip(y2,2,H-1)
            return bbox
        else:
            bbox[0] = np.clip(bbox[0],0,W-2)
            bbox[1] = np.clip(bbox[1],0,H-2)
            x2 = np.clip(x2,2,W-1)
            y2 = np.clip(y2,2,H-1)
            w,h = x2 - bbox[0], y2-bbox[1]
            bbox[2],bbox[3] = w,h
            return np.int32(bbox)
        
    def batch_clip_bbox(self,bbox,im_h,im_w):
        x2,y2 = bbox[:,0] + bbox[:,2], bbox[:,1] + bbox[:,3]
        bbox[:,0] =  torch.clip(bbox[:,0],0,im_w-2)
        bbox[:,1] = torch.clip(bbox[:,1],0,im_h-2)
        x2 = torch.clip(x2,2,im_w-1)
        y2 = torch.clip(y2,2,im_h-1)
        w,h = x2 - bbox[:,0], y2-bbox[:,1]
        bbox[:,2],bbox[:,3] = w,h
        return bbox
        
    def _get_jittered_historybox(self, boxs):
        """ Jitter the input box
        args:
            box - input bounding box (x,y,w,h)
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """
        box_list = []
        for box in boxs:
            jittered_size = box[:,2:4] * torch.abs(torch.randn(box.shape[0],2)+1).clamp(min=(0.8),max=(1.2))
            max_offset = (box[:,2:4] / 2) * torch.randn(box.shape[0],2).clamp(min=(-0.25),max=(0.25))
            jittered_center = box[:,0:2] + 0.5 * box[:,2:4] + max_offset * (torch.rand(box.shape[0],2) - 0.5)
            box_list.append(torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=1))
        return box_list
        
    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
                search has tow variable, 0 is as search, 1 is as dynamic
        """
        # Apply joint transforms
        N = len(data['search_images'])
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)
        
        

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data
            # cv2.imwrite('t.jpg',crops[1])
            # Crop image region centered at jittered_anno box and get the attention mask
            # self.draw_bbox(torch.tensor(np.stack(crops)).permute(0,3,1,2),torch.stack(boxes)*self.output_sz[s])
            if s =='search':
                # cv2.imwrite('x.jpg',crops[0])
                crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                                data[s + '_anno'], self.search_area_factor[s],
                                                                                self.output_sz[s], masks=data[s + '_masks'])

                imgs, boxes, data[s + '_att'], data[s + '_masks'] = self.transform[s](
                            image=[crops], bbox=[boxes], att=att_mask, mask=mask_crops, joint=False)

                data[s +'_anno'] = torch.stack(boxes[0],dim=0)
                data[ s + '_images'] = torch.stack(imgs[0],dim=0)
                data[s +'_anno'] = self.batch_clip_bbox_normal(data[s +'_anno'],self.output_sz[s])
                # self.draw_hisbbox(data[ s + '_images'][0][0],history_bbox*self.output_sz[s])
            else:
                crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[ s + '_images'], jittered_anno,
                                                                                data[s + '_anno'], self.search_area_factor[s],
                                                                                self.output_sz['template'], masks=data[s + '_masks'],
                                                                                )
                # Apply transforms
                data[ s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform['template'](
                    image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)
                data[s +'_anno'] = self.batch_clip_bbox_normal(torch.stack(data[s +'_anno'],dim=0),self.output_sz[s])
            # self.draw_bbox(data['template' + '_images'][0].unsqueeze(0),data['template' + '_anno'][0]*self.output_sz['template'])
            # if s == 'template':
            #     self.draw_bbox(torch.stack(data[ s + '_images'],dim=0),data[s + '_anno']*self.output_sz[s])
            # else:
            #     self.draw_bbox(data[ s + '_images'],data[s + '_anno']*self.output_sz[s])
            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        # utils.save_image(data['search_images'][0][1],'simg.jpg') utils.save_image(data['search_images'][0][0],'simg.jpg')
        # utils.save_image(data['template_images'][0][0],'timg.jpg')    utils.save_image(data['gt_label'][0],'gt_label.jpg')
        # utils.save_image(data['recover_frames'][0][0],'rimg.jpg')
        # utils.save_image(data['dyn_template_images'][0][0],'dtimg.jpg')
        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))


        data['gt_mask'],data['gt_gauss_mask'],data['score_ve_label'] = self.target_mask(data['search_anno'])
        data['template_lable'],_,_ = self.target_mask(data['template_anno'],mode='template')
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
        return data
    
    def create_dynamic_image(self,search,anno):
        N,C,H,W = search.shape
        conner_bbox = bbutils.batch_xywh2corner(anno*H).to(torch.int32)
        image_list = []
        for n in range(N-1):
            bbox = conner_bbox[n]
            mask = torch.zeros(1,H,W)
            dynamic_image = search[n].clone()
            mask[:,bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
            image_list.append(dynamic_image * mask)
        return torch.stack(image_list)
    
    def target_mask(self,bbox,mode='search',downsample=True):
        bboxs = (bbox.view(-1,4) * self.output_sz[mode])
        mask_list = []
        gauss_mask_lsit = []
        pure_gauss_mask_list = []
        for bbox in bboxs:
            mask = torch.zeros(self.output_sz[mode],self.output_sz[mode])
            bbox[2],bbox[3] = bbox[0] + bbox[2],bbox[1] + bbox[3]
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
            hw = self.output_sz[mode] // self.settings.stride
            mask = F.interpolate(mask,(hw,hw))
            gauss_mask = F.interpolate(gauss_mask,(hw,hw))
            pure_gauss_mask = F.interpolate(pure_gauss_mask,(hw,hw),mode='bilinear')
        return mask,gauss_mask,pure_gauss_mask
    
    def draw_bbox(self,imgs_,bboxs):
        imgs = imgs_.clone()
        mean = torch.tensor((0.4379, 0.4251, 0.4685)).reshape(1,1,1,3).to(imgs)
        std = torch.tensor((0.2361, 0.2416, 0.2415)).reshape(1,1,1,3).to(imgs)
        imgs = ((imgs.permute(0,2,3,1) *std + mean) * 255).to(torch.int32).contiguous()
        bboxs = bboxs.view(-1,4)
        id = 1
        for img, bbox in zip(imgs,bboxs):
            img = img.detach().cpu().numpy()
            bbox = bbox.to(torch.int32).tolist()
            img = cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,0,255),1)
            cv2.imwrite('MemTrack/visual_gt_bbox/'+str(id)+'.jpg',img)

    def draw_hisbbox(self,img_,bboxs):  # b c h w ; b l 4
        for i in range(bbox.shape[0]):
            img = img_[i].clone()
            bbox_ = bboxs[i]
            mean = torch.tensor((0.4379, 0.4251, 0.4685)).reshape(1,1,3).to(img)
            std = torch.tensor((0.2361, 0.2416, 0.2415)).reshape(1,1,3).to(img)
            img = ((img.permute(1,2,0) *std + mean) * 255).to(torch.int32).contiguous()
            img = img.detach().cpu().numpy()
            bbox_ = bbox_.view(-1,4) * 256
            id = 0
            for bbox in bbox_:
                id += 1
                bbox = bbox.to(torch.int32).tolist()
                img = cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),1)
                cv2.imwrite('MemTrack/visual_his_bbox/'+str(id)+'.jpg',img) 

        
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
    
    def vis_mask(self,mask,gaussian_masked,center_y,center_x,sigma,interpolate_gaussian_masked):
        H,W = mask.shape
        plt.figure(figsize=(15, 5))
        # Plot mask
        plt.subplot(1, 4, 1)
        plt.title('Mask')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        # Plot Gaussian distribution
        plt.subplot(1, 4, 2)
        plt.title('Gaussian')
        cax = plt.imshow(torch.exp(-0.5 * ((torch.arange(H).float().unsqueeze(1).expand(H, W) - center_y.float())**2 +
                                            (torch.arange(W).float().unsqueeze(0).expand(H, W) - center_x.float())**2) / sigma**2).float(),
                        cmap='viridis')
        plt.colorbar(cax, ax=plt.gca(), orientation='vertical', label='Value')
        plt.axis('off')

        # Plot Masked Gaussian
        plt.subplot(1, 4, 3)
        plt.title('Masked Gaussian')
        cax = plt.imshow(gaussian_masked, cmap='viridis')
        plt.colorbar(cax, ax=plt.gca(), orientation='vertical', label='Value')
        plt.axis('off')

        # Plot interpolate Masked Gaussian
        plt.subplot(1, 4, 4)
        plt.title('interpolate Masked Gaussian')
        cax = plt.imshow(interpolate_gaussian_masked.squeeze(), cmap='viridis')
        plt.colorbar(cax, ax=plt.gca(), orientation='vertical', label='Value')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('mask.png')

def transform_image(image, do_flip):
        if do_flip:
            if torch.is_tensor(image):
                return image.flip((3,))
            return np.fliplr(image).copy()
        return image


def vis(img1,bbox):
    """_summary_

    Args:
        img (np.array): h,w,c
        bbox (list): x,y,w,h
    """
    img = img1.copy()
    for i in range(img.shape[2]):
        img[:,:,i] = ((img[:,:,i] - img[:,:,i].min()) / (img[:,:,i].max()-img[:,:,i].min())) * 255
    img = img.astype(np.int32)
    x1,y1 = bbox[:2]
    x2,y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
    img = img[:,:,:3]
    img = cv.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
    cv.imwrite('simg_bbox.jpg',img)

    
    