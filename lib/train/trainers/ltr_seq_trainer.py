import os
import datetime
from collections import OrderedDict
from torch.nn.utils import clip_grad_norm_
from lib.train.data.wandb_logger import WandbWriter
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from memory_profiler import profile
from lib.train.admin import TensorboardWriter
import torch
import time
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from lib.utils.misc import get_world_size
import cv2

class LTRSeqTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False,tb_writer=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard and wandb
        self.wandb_writer = None
        if settings.local_rank in [-1, 0]:
           tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
           if not os.path.exists(tensorboard_writer_dir):
               os.makedirs(tensorboard_writer_dir)
           self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

           if settings.use_wandb:
               world_size = get_world_size()
               cur_train_samples = self.loaders[0].dataset.samples_per_epoch * max(0, self.epoch - 1)
               interval = (world_size * settings.batchsize)  # * interval
               self.wandb_writer = WandbWriter(settings.project_path[6:], {}, tensorboard_writer_dir, cur_train_samples, interval)

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        print("move_data", self.move_data_to_gpu)
        self.settings = settings
        self.use_amp = use_amp
        self.tb_writer = tb_writer
        if use_amp:
            self.scaler = GradScaler()

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

        self.miou_list = []

    def draw_bbox(self,imgs,bboxs,sv_path):
        """
        bboxs: x,y,w,h
        """
        id = 0
        for img, bbox in zip(imgs,bboxs):
            id += 1
            bbox = bbox.to(torch.int32).tolist()
            if img.shape[2] != 3:
                img = (img[:,:,:3] * 255).copy()
            img = cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,0,255),1)
            cv2.imwrite(sv_path+str(id)+'.jpg',img)

    def cycle_dataset(self, loader,):
        """Do a cycle of training or validation."""
        torch.autograd.set_detect_anomaly(True)
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()
        # cv2.imwrite('simg.jpg',data['rgb_search_images'][0][1])
        for i, data in enumerate(loader, 1):
            self.actor.eval()
            self.data_read_done_time = time.time()

            tb_idx = i + (self.epoch-1) * loader.__len__()

            # self.draw_bbox(data['hp_template_images'],torch.stack(*data['template_annos']),'/home/zhoujiawei/HOTchanllenge/template_bbox/')
            # self.draw_bbox(data['hp_search_images'][1],data['search_annos'][1],'/home/zhoujiawei/HOTchanllenge/search_bbox/')
            with torch.no_grad():
                explore_result = self.actor.explore(data)   # 为了得到pre_sqe作为历史信息预测
            if explore_result == None:
                print("this time i skip")
                continue
            # get inputs
            
            self.data_to_gpu_time = time.time()

            data['epoch'] = self.epoch
            data['settings'] = self.settings

            stats = {}

            miou_record = []    # batch个视频的所有序列平均iou

            num_seq = len(data['template_images'])
            num_frames = len(data['search_images'][0])-1
            # Calculate reward tensor

            baseline_iou = explore_result['baseline_iou']

            for seq_idx in range(num_seq):
                b_miou = torch.mean(baseline_iou[:, seq_idx]) # 每一个视频的平均iou
                miou_record.append(b_miou.item())
                b_reward = b_miou.item()

            # Training mode 训练少一张搜索图片
            bs_backward = 1
            replace_index = [explore_result['replace_index'][n][b] for b in range(num_seq) for n in range(num_frames)]
            replace_index = [replace_index[b*num_frames:b*num_frames+num_frames] for b in range(num_seq)]
            # print(self.actor.net.module.box_head.decoder.layers[2].mlpx.fc1.weight)
            # self.actor.train(True)
            if loader.training:
                self.optimizer.zero_grad()
            for bs in range(num_seq): # 分别取出每一个视频的变量
                # print("now is ", cursor , "and all is ", num_seq)
                model_inputs = {}
                model_inputs['template_images'] = explore_result['template_images'][bs:bs+1].cuda()   # 模板图像z_crop
                model_inputs['template_label'] = explore_result['template_label'][bs:bs+1].cuda()
                model_inputs['box_mask_z'] = explore_result['box_mask_z'][bs:bs+1].cuda()
                model_inputs['search_images'] = explore_result['search_images'][bs].cuda()  # 搜索图像x_crop
                model_inputs['search_anno'] = explore_result['search_anno'][bs].cuda()  # search图像中真实框
                model_inputs['pred_label'] = explore_result['pred_label'][bs,1:].cuda()
                model_inputs['gt_mask'] = explore_result['gt_mask'][bs].cuda()
                model_inputs['gt_gauss_mask'] = explore_result['gt_gauss_mask'][bs].cuda()
                model_inputs['score_ve_label'] = explore_result['score_ve_label'][bs].cuda()
                model_inputs['update_index'] = explore_result['update_index'][bs,:-1].cuda()
                model_inputs['replace_index'] = replace_index[bs][:-1]
                model_inputs['epoch'] = data['epoch']

                loss, stats_cur = self.actor.compute_sequence_losses(model_inputs)
                # for name, param in self.actor.net.named_parameters():
                #    shape, c = (param.grad.shape, param.grad.sum()) if param.grad is not None else (None, None)
                #    print(f'{name}: {param.shape} \n\t grad: {shape} \n\t {c}')
                # print("i make this!")
                if loader.training:
                    loss.backward() # 每次计算一个视频序列的梯度
                # print("i made that?")

                for key, val in stats_cur.items():
                    if key in stats:
                        stats[key] += val * (bs_backward / num_seq)
                    else:
                        stats[key] = val * (bs_backward / num_seq)


            if loader.training:
                if self.settings.grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(),self.settings.grad_clip_norm)
                # print(self.actor.net.module.backbone.blocks[8].mlp.fc1.weight)
                self.optimizer.step()  
                # print(self.optimizer)
                stats['grad_norm'] = grad_norm.detach().cpu().item()
            miou = np.mean(miou_record)
            self.miou_list.append(miou)
            stats['mIoU'] = miou
            stats['mIoU10'] = np.mean(self.miou_list[-10:])
            stats['mIoU100'] = np.mean(self.miou_list[-100:])

            batch_size = num_seq * np.max(data['search_images'][0].shape[0])
            self._update_stats(stats, batch_size, loader)
            self._print_stats(i, loader, batch_size)
            torch.cuda.empty_cache()

            if self.settings.local_rank in [-1,0] and self.tb_writer is not None:
                if 'train' in loader.name:
                    for k, v in self.stats['train'].items():
                        if 'LearningRate' not in k:
                            v = v.avg
                            self.tb_writer.add_scalar('train_{}'.format(k),v,tb_idx)
                else:
                    for key, value in self.stats['val'].items():
                        if 'LearningRate' not in key:
                            value = value.avg
                            self.tb_writer.add_scalar('val_{}'.format(key),value,tb_idx)

            # update wandb status
            if self.wandb_writer is not None and i % self.settings.print_interval == 0:
               if self.settings.local_rank in [-1, 0]:
                   self.wandb_writer.write_log(self.stats, self.epoch)

        # calculate ETA after every epoch
        epoch_time = self.prev_time - self.start_time
        print("Epoch Time: " + str(datetime.timedelta(seconds=epoch_time)))
        print("Avg Data Time: %.5f" % (self.avg_date_time / self.num_frames * batch_size))
        print("Avg GPU Trans Time: %.5f" % (self.avg_gpu_trans_time / self.num_frames * batch_size))
        print("Avg Forward Time: %.5f" % (self.avg_forward_time / self.num_frames * batch_size))

    def train_epoch(self,):
        """Do one epoch for each loader."""
        if self.epoch == 40:
            self.loaders[1].epoch_interval = 1
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader,)

        self._stats_new_epoch()
        # if self.settings.local_rank in [-1, 0]:
        #    self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        # add lr state
        if loader.training:
            lr_list = self.lr_scheduler.get_last_lr()
            for i, lr in enumerate(lr_list):
                var_name = 'LearningRate/group{}'.format(i)
                if var_name not in self.stats[loader.name].keys():
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        prev_frame_time_backup = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += (self.data_read_done_time - prev_frame_time_backup)
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)

            # 2021.12.14 add data time print
            print_str += 'DataTime: %.3f (%.3f)  ,  ' % (
            self.avg_date_time / self.num_frames * batch_size, self.avg_gpu_trans_time / self.num_frames * batch_size)
            print_str += 'ForwardTime: %.3f  ,  ' % (self.avg_forward_time / self.num_frames * batch_size)
            print_str += 'TotalTime: %.3f  ,  ' % ((current_time - self.start_time) / self.num_frames * batch_size)
            # print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.data_read_done_time - prev_frame_time_backup, self.data_to_gpu_time - self.data_read_done_time)
            # print_str += 'ForwardTime: %.3f  ,  ' % (current_time - self.data_to_gpu_time)
            # print_str += 'TotalTime: %.3f  ,  ' % (current_time - prev_frame_time_backup)

            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    # def _write_tensorboard(self):
    #    if self.epoch == 1:
    #        self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

    #    self.tensorboard_writer.write_epoch(self.stats, self.epoch)

