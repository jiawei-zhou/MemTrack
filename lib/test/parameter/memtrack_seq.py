from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.memtrack_seq.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir    # get value in local.py 
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/memtrack_seq/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)  # the infomation of cfg is updated by the yaml file 
    params.cfg = cfg
    print("test config: ", cfg)
    print("test epoch: ", cfg.TEST.EPOCH)
    print("test tp_num: ", cfg.TEST.TP_NUM)
    print("test memory_num: ", cfg.MODEL.MEMORYBLOCK.MEMORY_NUM)
    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path 
    params.checkpoint = os.path.join(save_dir, "checkpoints/train/memtrack_seq/%s/MemTrackSeq_ep%04d.pth.tar" %
                                     (yaml_name, cfg.TEST.EPOCH))

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
