import numpy as np
import sys
sys.path.append('/home/zhoujiawei/my_model/MemTrack')
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import json  
import os
class VISODataset(BaseDataset):

    def __init__(self): # self由BaseDatset传入
        super().__init__()
        self.base_path = self.env_settings.viso_path     # 数据集地址
        with open(os.path.join(self.base_path, 'VISO_test.json'), 'r') as f:
            self.sequence_info_list = json.load(f)  # 数据集json文件

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in list(self.sequence_info_list.keys())])
    
    def _construct_sequence(self, name):
        sub_dict = list(self.sequence_info_list[name].keys())
        attr,gt_rect,img_names,init_rect,video_dir = sub_dict[0], sub_dict[1], sub_dict[2], sub_dict[3], sub_dict[4]
        frame = [os.path.join(self.base_path,path) for path in self.sequence_info_list[name][img_names]]
        ground_truth = np.array(self.sequence_info_list[name][gt_rect])
        return Sequence(name,frame,'VISO',ground_truth)
    
    def __len__(self):
        return len(self.sequence_info_list)

if __name__ == '__main__':
    data = VISODataset()
    a = data.get_sequence_list()