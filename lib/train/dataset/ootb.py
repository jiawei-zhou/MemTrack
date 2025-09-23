import os
import os.path
import torch
import numpy as np
import pandas
# import csv
import random
# import sys
# sys.path.append('/home/zhoujiawei/my_model/History_tracking')
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
# from base_video_dataset import BaseVideoDataset
import cv2
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
import json
class OOTB(BaseVideoDataset):
    """
    VastTrack test set consisting of 3500 videos

    Publication:
        VastTrack: Vast Category Visual Object Tracking
        Liang Peng, Junyuan Gao, Xinran Liu, Weihong Li, Shaohua Dong, Zhipeng Zhang, Heng Fan and Libo Zhang
        https://arxiv.org/pdf/2403.03493.pdf

    Download the dataset from https://github.com/HengLan/VastTrack
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None):

        root = env_settings().ootb if root is None else root
        super().__init__('OOTB', root, image_loader)

        with open(os.path.join(root, 'OOTB_new.json'), 'r') as f:
            self.meta_data = json.load(f)
        # Keep a list of all classes
        self.class_list = ['car','shipe','plane','train']
        self.class_to_id = {classe: i for i,classe in enumerate(self.class_list)}

        self.sequence_list = [sequence for sequence in self.meta_data.keys()]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.seq_per_class = self._build_class_list()

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class
    
    def get_attr(self,seq_id):
        seq_name = self.sequence_list[seq_id]
        attr = self.meta_data[seq_name]['attr']
        return attr
    
    def get_name(self):
        return 'OOTB'

    def has_class_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)
    
    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "new_groundtruth_hbb.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',',dtype=np.float32, header=None, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        class_name = seq_name.split('_')[0]
        vid_id = seq_name.split('_')[1]

        return os.path.join(self.root, class_name + '_' + vid_id)
    
    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img', '{:04}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-1].split('_')[0]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)
        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

if __name__ =='__main__':
    root = '/home/zhoujiawei/satellite_video_datasets/OOTB'
    dataset = OOTB(root=root,split='train')
    num = 0
    base_path = '/home/zhoujiawei/my_model/test_ootb'
    for i in range(len(dataset.sequence_list)):
        info = dataset.get_sequence_info(i)
        indexs = [i for i in range(len(info['bbox']))]
        index = random.choices(indexs,k=1)
        frame_list, anno_frames,_ = dataset.get_frames(i,index)
        base_path1 = os.path.join(base_path,dataset.sequence_list[i])
        if not os.path.exists(base_path1):
            os.makedirs(base_path1)
        for frame,anno,idx in zip(frame_list,anno_frames['bbox'],index):
            gt_bbox = anno.int().tolist()
            gt_bbox[2],gt_bbox[3] = gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]
            frame = cv2.rectangle(frame,(gt_bbox[0],gt_bbox[1]),(gt_bbox[2],gt_bbox[3]),(0,0,255),1)
            sv_path = os.path.join(base_path1,str(idx)+'img.jpg')
            cv2.imwrite(sv_path,frame)
        print(i,'is checked')
    print(num)
    