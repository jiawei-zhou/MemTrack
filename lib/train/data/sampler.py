import random
import torch.utils.data
from lib.utils import TensorDict
import numpy as np
import cv2
import time
import math
def no_processing(data):
    return data


class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. 

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing, frame_sample_mode='causal',
                 train_cls=False, pos_prob=0.5,pre_num=24,template_gap=24,dyn_template=True,search_factor=3,reverse_prob=0.5,sample_max_gap=24):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets
        self.train_cls = train_cls  # whether we are training classification
        self.pos_prob = pos_prob  # probability of sampling positive class when making classification

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]


        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode
        self.template_gap  = template_gap 
        self.pre_num = pre_num
        self.reverse_prob = reverse_prob
        self.sample_gap = np.linspace(1,sample_max_gap,sample_max_gap).astype(np.int32)
        self.static_nums = np.linspace(1,pre_num,pre_num).astype(np.int32)

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False,state=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return None
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
               
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0 or len(valid_ids) < num_ids:
            return None
        ids = random.choices(valid_ids, k=num_ids)
        ids = sorted(ids)
        return ids
    
    def sample_history_ids(self,valid_ids,search_frames_ids,mode='order',static=False):
        history_ids_list = []
        valid_ids = np.array(valid_ids)
        for search_frames_id in search_frames_ids:
            if search_frames_id not in valid_ids:
                print(search_frames_id)
                print(valid_ids)
            id_index = np.where(valid_ids==search_frames_id)[0][0]
            if mode == 'order':
                if static:
                    nums = random.choices(self.static_nums,k=1)[0]
                    res_nums = self.pre_num - nums
                    min_index = max(0,id_index-res_nums-1)
                    history_ids = [valid_ids[min_index]] * nums
                    history_ids = history_ids + list(valid_ids[min_index+1:id_index])
                else:
                    min_index = max(0,id_index-self.pre_num)
                    history_ids = list(valid_ids[min_index:id_index])
                while len(history_ids) < self.pre_num:
                    history_ids.append(valid_ids[0])
            elif mode == 'reversed':
                if static:
                    nums = random.choices(self.static_nums,k=1)[0]
                    res_nums = self.pre_num - nums
                    max_index = min(id_index+1+res_nums,len(valid_ids)-1)
                    history_ids = [valid_ids[max_index]] * nums
                    history_ids = history_ids + list(valid_ids[id_index+1:max_index])
                else:
                    max_index = min(id_index+1+self.pre_num,len(valid_ids)-1)
                    history_ids = list(valid_ids[id_index+1:max_index])
                while len(history_ids) < self.pre_num:
                    history_ids.append(valid_ids[-1])
            history_ids = sorted(history_ids)
            history_ids_list.append(history_ids)

        return history_ids_list
    
    
    def static_sample_ids(self,visible,search_frame_ids):
        val = False
        gap = 1
        while not val:
            if visible[search_frame_ids-gap]:
                history_ids = [search_frame_ids-gap] * self.pre_num
                val = True
            gap += 1
        return history_ids
    
    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict
    
    def clip_bbox(self,bbox,im_h,im_w):
        x2,y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
        bbox[0] = torch.clip(bbox[0],0,im_w-2)
        bbox[1] = torch.clip(bbox[1],0,im_h-2)
        x2 = torch.clip(x2,2,im_w-1)
        y2 = torch.clip(y2,2,im_h-1)
        w,h = x2 - bbox[0], y2-bbox[1]
        bbox[2],bbox[3] = w,h
        return bbox
    
    def batch_clip_bbox(self,bbox,im_h,im_w):
        x2,y2 = bbox[:,0] + bbox[:,2], bbox[:,1] + bbox[:,3]
        bbox[:,0] =  torch.clip(bbox[:,0],0,im_w-2)
        bbox[:,1] = torch.clip(bbox[:,1],0,im_h-2)
        x2 = torch.clip(x2,2,im_w-1)
        y2 = torch.clip(y2,2,im_h-1)
        w,h = x2 - bbox[:,0], y2-bbox[:,1]
        bbox[:,2],bbox[:,3] = w,h
        return bbox
    
    def __getitem__(self, index):
        if self.train_cls:
            return self.getitem_cls()
        else:
            return self.getitem()

    def getitem(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """

        valid = False
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0
                # try:
                if random.random() < self.reverse_prob:
                    order_sample = True
                else:
                    order_sample = False
                if self.frame_sample_mode == 'causal':
                    if order_sample:
                    # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                        while search_frame_ids is None:
                            base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                    max_id=len(visible) - self.num_search_frames - gap_increase)
 
                            template_frame_ids = base_frame_id 
                            # add two sample nums used as static history bbox
                            search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                    max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                    num_ids=self.num_search_frames,state='search')
                            # Increase gap until a frame is found
                            gap_increase += 5
                    else:   # sample frames id in reversed, i.e template frame id > search frames id
                        while search_frame_ids is None:
                            base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_search_frames + 1 + gap_increase,
                                                                    max_id=len(visible)- self.num_template_frames)

                            template_frame_ids = base_frame_id 
                            # add two sample nums used as static history bbox
                            search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0]-self.max_gap,
                                                                    max_id=template_frame_ids[0] -1,
                                                                    num_ids=self.num_search_frames,state='search')
                            # Increase gap until a frame is found
                            gap_increase += 5
                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")

            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames 
            
            try:
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)  # 0是base 1是pre 
                search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                try:
                    np.stack(search_frames)
                except:
                    continue
                H, W, _ = template_frames[0].shape
                for i in range(len(search_anno['bbox'])):
                    search_anno['bbox'][i] = self.clip_bbox(search_anno['bbox'][i],H,W)
                
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros((H, W))] * self.num_template_frames
                search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros((H, W))] * self.num_search_frames

                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})
                # make data augmentation
                data = self.processing(data)
                # check whether data is valid
                valid = data['valid']
            except Exception  as e:
                print(e)
                valid = False
                # raise ValueError(seq_id,'Sample getting error')
        return data

def draw_tmbbox(imgs_,bboxs):
    id = 0
    for img_, bbox in zip(imgs_,bboxs):
        id += 1
        img = img_.copy()
        bbox = bbox.to(torch.int32).tolist()
        img = cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,0,255),1)
        cv2.imwrite('/home/zhoujiawei/my_model/HisTrack/visual_tm_bbox/'+str(id)+'.jpg',img)

def draw_shbbox(imgs_,bboxs):
    id = 0
    for img_, bbox in zip(imgs_,bboxs):
        id += 1
        img = img_.copy()
        bbox = bbox.to(torch.int32).tolist()
        img = cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,0,255),1)
        cv2.imwrite('/home/zhoujiawei/my_model/HisTrack/visual_gt_bbox/'+str(id)+'.jpg',img)

def draw_hisbbox(imgs_,bboxs):
    bboxs = bboxs.view(-1,4) 
    id = 0
    for img_  in imgs_:
        id += 1
        img = img_.copy()
        for bbox in bboxs:
            bbox = bbox.to(torch.int32).tolist()
            img = cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,0,255),1)
            cv2.imwrite('/home/zhoujiawei/my_model/HisTrack/visual_his_bbox/'+str(id)+'.jpg',img) 