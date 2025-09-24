# MemTrack: Utilizing the Temporal Memory Transformer Network for Satellite Video Object Tracking
* # [ISPRS] - Incorporating prior knowledge and temporal memory transformer network for satellite video object tracking
> [**Incorporating prior knowledge and temporal memory transformer network for satellite video object tracking**](https://www.sciencedirect.com/science/article/pii/S0924271625003028)<br>
Models:[[google drive](https://drive.google.com/file/d/1GiadyYZxFZOO-nzqpiNemrz6rUrJH6kB/view?usp=sharing),[baiduyun](https://pan.baidu.com/s/1Qjw4NFHkmPyl12eSncij6g?pwd=ckky)] , raw results:[[google drive](https://drive.google.com/drive/folders/1pAvzBrjJLt8A4ujOW9VwxKhE6DAbQbG6?usp=sharing),[baiduyun](https://pan.baidu.com/s/1wG9kpv0w69YKXTYHAqPFPA?pwd=tive)] <br>

The MemTrack framework is as follows:
![image](framework.jpg)
Fig. 1. The overall framework of the proposed MemTrack.  <br>

Experimental results demonstrate the superior performance of our method. <br>

Table. 1. Performance comparison of trackers across different datasets. The best results are highlighted in red, the second-place with blue and the third-place with green.
![image](table_result.png)

# 1. Dataset preparation
The testing datasets are avalible in:[SatSOT](http://www.csu.cas.cn/gb/kybm/sjlyzx/gcxx_sjj/sjj_wxxl/202106/t20210607_6080256.html)，[SV248S](https://github.com/xdai-dlgvv/SV248S)，[OOTB](https://github.com/YZCU/OOTB) and [VISO](https://github.com/QingyongHu/VISO), which are all large public real satellite video datasets. Due to lack of offical json files (SV248S,OOTB and VISO), we create the specifical `json files` for datasets, which can be donwload in [baiduyun](https://pan.baidu.com/s/163glyhr5LYR8HC62Ueiy8Q?pwd=jn7c)(code:jn7c) or [google driver](https://drive.google.com/file/d/1T5T77KByGbBTX06uN4_KP4WoZLdDkRxA/view?usp=sharing) and the corresponding `dataset.py` files. Put the json files into coresponding dataset path.
```python 
--SV248S
  -- 01
  -- ...
  -- 06
  -- SV248S_new.json
--OOTB
  -- anno
  -- car_01
  -- ...
  -- OOTB_new.json
--VISO
  --SOT
    --sot
      --VISO_test.json
```
<br>
The training dataset contains four commonly used datasets for general video object tracking in the field of computer vision, which can be easily obtained from the official websites of their respective datasets.
```python 
GOT10K
LaSOT
TrackingNet
WebUAV3M
```

# 2. Install the environment
In this work, we use python=3.10 and torch==2.3.1+cu118 or torch==2.3.1+cu12.1. The specific installation packages are listed in conda_environment.yaml, which can be installed in the following way:
> Install the environment option1 torch==2.3.1+cu118
```python 
modified prefix path in conda_environment
conda env create -f conda_environment.yml
```
Or manually create the Python environment, and then install other packages，which can be installed in the following way:
> Install the environment option2 torch==2.3.1+cu12.1
```python 
conda create -n MemTrack python=3.10
conda activate MemTrack
pip install -r requirements.txt
```
## Set project paths
Run the following command to set paths for this project
> Set project paths
```python 
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
> Modify the paths
```python 
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

# 3. Training
Download [pre-trained weights](https://drive.google.com/file/d/1l0YSK0QLPGVIGiNXwlaWp5uhIkJawJqh/view) and put it under [./pretrained](pretrained) <br>

> run one stage training
```python 
python tracking/train.py --script memtrack --config memtrack_384_full --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```

> run two stage training
```python 
1. modified the PRETRAIN_PTH in [yaml file](experiments/memtrack_seq/memtrack_seq_384_full.yaml)
2. python tracking/train.py --script memtrack --config memtrack_384_full --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```

# 4. Test and evaluate on benchmarks
Put the downloaded checkpoints under [./output/checkpoints/train/memtrack_seq/memtrack_seq_384_full](/output/checkpoints/train/memtrack_seq/memtrack_seq_384_full)

> run testing
```
python tracking/test.py 
-- tracker_name memtrack_seq  
-- tracker_param memtrack_seq_384_full 
-- dataset satsot sv248s viso ootb 
-- threads 1
-- num_gpu 1
-- debug 0
-- tracker_cls None # different tracker [without_tp,kalman]
-- save_name None # results saving name
-- sequence None # test sequences in the specified dataset
```

> run evaluate
```
1. modified the name, parameter_name and dataset name in analysis_results
2. python tracking/analysis_results.py
```

> test FLOPs, Params and Speed
```
python tracking/profile_model.py --script memtrack_seq --config memtrack_384_full
```

# 5. Acknowledgments
- Thanks for the [DropTrack](https://github.com/jimmy-dq/DropTrack) and [pytracking](https://github.com/visionml/pytracking) libraries for convenient implementation.

# 6. Citation
```
@article{ZHOU2025630,
title = {Incorporating prior knowledge and temporal memory transformer network for satellite video object tracking},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
author = {Jiawei Zhou and Yanni Dong and Yuxiang Zhang and Bo Du},
volume = {228},
pages = {630-647},
year = {2025},
issn = {0924-2716}}
```