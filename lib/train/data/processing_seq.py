import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F
import cv2
import numpy as np
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

    def __init__(self,mode='pair', settings=None, *args, **kwargs):
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
        self.mode = mode
        self.settings = settings
        self.output_sz = settings.output_sz

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms


        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'] , data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'] , mask=data['search_masks'], new_roll=False)
        data['template_images'] = np.stack(data['template_images'])
        data['search_images'] = np.stack(data['search_images'])
        data['valid']=True
        return data
    
    def target_mask(self,bbox,mode='search',downsample=True):
        bboxs = (bbox.view(-1,4) * self.output_sz[mode])
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
            hw = self.output_sz[mode] // self.settings.stride
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