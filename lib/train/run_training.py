import os
import sys
import argparse
import importlib
import cv2 as cv
import torch.backends.cudnn
import torch.distributed as dist
import torch
import random
import numpy as np
torch.backends.cudnn.benchmark = False

import _init_paths
import lib.train.admin.settings as ws_settings


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    cv.setNumThreads(1)
    cv.ocl.setUseOpenCL(False)


def run_training(script_name, config_name, cudnn_benchmark=True, local_rank=-1, save_dir=None, base_seed=None,
                 use_lmdb=False, script_name_prv=None, config_name_prv=None, use_wandb=False,use_tb=None,
                 distill=None, script_teacher=None, config_teacher=None):
    """Run the train script.
    args:
        script_name: Name of emperiment in the "experiments/" folder.
        config_name: Name of the yaml file in the "experiments/<script_name>".
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """
    if save_dir is None:
        print("save_dir dir is not given. Use the default dir instead.")
    # This is needed to avoid strange crashes related to opencv
    torch.set_num_threads(4)
    cv.setNumThreads(4)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('script_name: {}.py  config_name: {}.yaml'.format(script_name, config_name))

    '''2021.1.5 set seed for different process'''
    if base_seed is not None:
        if local_rank != -1:
            init_seeds(base_seed + local_rank)
        else:
            init_seeds(base_seed)

    settings = ws_settings.Settings()   
    settings.script_name = script_name  
    settings.config_name = config_name  
    settings.project_path = 'train/{}/{}'.format(script_name, config_name) 
    if script_name_prv is not None and config_name_prv is not None: 
        settings.project_path_prv = 'train/{}/{}'.format(script_name_prv, config_name_prv)
    settings.local_rank = local_rank
    settings.save_dir = os.path.abspath(save_dir)
    settings.use_lmdb = use_lmdb
    settings.use_tb = use_tb   
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    settings.cfg_file = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (script_name, config_name)) 
    settings.use_wandb = use_wandb
    if distill:
        settings.distill = distill
        settings.script_teacher = script_teacher
        settings.config_teacher = config_teacher
        if script_teacher is not None and config_teacher is not None:
            settings.project_path_teacher = 'train/{}/{}'.format(script_teacher, config_teacher)
        settings.cfg_file_teacher = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (script_teacher, config_teacher))
        expr_module = importlib.import_module('lib.train.train_script_distill')
    else:
        expr_module = importlib.import_module('lib.train.train_script')
    expr_func = getattr(expr_module, 'run')

    expr_func(settings)


def main():
    
    # debug 配置
    os.environ['CUDA_VISIBLE_DEVICES'] = '7,6,4,3'
    script = 'memtrack_seq'
    config = 'memtrack_seq_384_full'
    save_dir = './output'
    nproc_per_node = 1
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str,default=script, help='training script name')
    parser.add_argument('--config', type=str, default=config, help='yaml configure file name')
    parser.add_argument('--save_dir', type=str,default=save_dir, help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple", "multi_node"], default="multiple",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int,default=nproc_per_node, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, help='training script name')
    parser.add_argument('--config_prv', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)  # whether to use wandb
    # for knowledge distillation
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    parser.add_argument('--script_teacher', type=str, help='teacher script name')
    parser.add_argument('--config_teacher', type=str, help='teacher yaml configure file name')
    parser.add_argument('--use_tb', type=int, choices=[0, 1], default=0)  # whether to use tensorboard
    # for multiple machines
    parser.add_argument('--rank', type=int, help='Rank of the current process.')
    parser.add_argument('--world-size', type=int, help='Number of processes participating in the job.')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP of the current rank 0.')
    parser.add_argument('--port', type=int, default='20000', help='Port of the current rank 0.')

    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=42, help='seed for random numbers')

    args = parser.parse_args()
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)
    run_training(args.script, args.config, cudnn_benchmark=args.cudnn_benchmark,
                 local_rank=args.local_rank, save_dir=args.save_dir, base_seed=args.seed,
                 use_lmdb=args.use_lmdb, script_name_prv=args.script_prv, config_name_prv=args.config_prv,
                 use_wandb=args.use_wandb,use_tb=args.use_tb,
                 distill=args.distill, script_teacher=args.script_teacher, config_teacher=args.config_teacher)


if __name__ == '__main__':
    main()
