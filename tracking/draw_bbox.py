import _init_paths
from lib.test.evaluation import get_dataset, trackerlist
from lib.test.analysis.draw_utils import color_line
from tqdm import tqdm
import numpy as np
import os
import cv2
import lib.train.data.bounding_box_utils as bbutils
import multiprocessing
from itertools import product
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


only_gt = True  
threads = 1
parameter_name = 'gt'   
base_path = '/MemTrack/vis_bbox/' + parameter_name
trackers = []
dataset_name = 'viso'
tracker_name = 'memtrack_seq'
# track_names = ['Mixformer','SeqTrack','SiamBAN','SiamCAR']
trackers.extend(trackerlist(name=tracker_name, parameter_name=parameter_name, dataset_name=dataset_name,
                            run_ids=None, display_name=tracker_name))
# for name in track_names:
#     trackers.extend(trackerlist(name=name, parameter_name=name, dataset_name=dataset_name,
#                             run_ids=None, display_name=name))
    
base_path = os.path.join(base_path,dataset_name)

color = color_line(trackers)
for key, v in color.items():
    temp = (v[2]*255,v[1]*255,v[0]*255)
    color[key] = temp
dataset = get_dataset(dataset_name)
trks_pred = {}

def draw_bbox(seq):
    anno_bb = np.array((seq.ground_truth_rect)).astype(np.int32)
    anno_bb = bbutils.batch_xywh2corner(anno_bb)
    for trk_id, trk in enumerate(trackers):
        # Load results
        base_results_path = '{}/{}/{}'.format(trk.results_dir,seq.dataset,seq.name)
        results_path = '{}.txt'.format(base_results_path)
        if os.path.isfile(results_path):
            try:
                pred_bb = np.loadtxt(str(results_path), delimiter=',',).astype(np.int32)
            except:
                pred_bb = np.loadtxt(str(results_path), delimiter='\t').astype(np.int32)
            trks_pred[trk.display_name] = bbutils.batch_xywh2corner(pred_bb)
            if len(anno_bb) != 1:
                assert len(trks_pred[trk.display_name]) == len(anno_bb)
    base_path_ = os.path.join(base_path,seq.name)
    if not os.path.exists(base_path_):
        os.makedirs(base_path_)
    for id , img_path in enumerate(seq.frames):
        im_name = img_path.split('/')[-1].split('.')[0]
        save_path = os.path.join(base_path_,im_name + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if len(anno_bb) != 1:
            gt_bbox = anno_bb[id]
            img = cv2.rectangle(img,(gt_bbox[0],gt_bbox[1]),(gt_bbox[2],gt_bbox[3]),(0,233,255),1)
        if not only_gt:
            for trk_name,trk in trks_pred.items():
                bbox = trk[id]
                img = cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color[trk_name],1)
        cv2.imwrite(save_path,img)

def print_error(value):
    print("error: ", value)

def process_with_multiprocessing(seqs, threads):
    seqs = [seq for seq in product(dataset)]
    with multiprocessing.Pool(threads) as pool:
        with tqdm(total=len(seqs)) as pbar:
            def update(*a):
                pbar.update(1)
            for seq in seqs:
                pool.apply_async(draw_bbox, args=seq, callback=update,error_callback=print_error)
            pool.close()
            pool.join()

start_time = time.time()
process_with_multiprocessing(dataset,threads)
end_time = time.time()

done_time = (end_time-start_time)
print('consume time:{:.1f}s done'.format(done_time))