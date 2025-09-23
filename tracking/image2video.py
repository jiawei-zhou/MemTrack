import cv2
import os
import multiprocessing
import time
from tqdm import tqdm

def images_to_video(image_folder, output_video, fps):
    # 获取图像文件列表，并按文件名排序
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()

    # 读取第一张图像以获取图像尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 定义视频编码器，设置输出视频文件、帧率和视频尺寸
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # mp4 格式编码器
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)  # 将图像帧写入视频

    video.release()  # 释放视频写入对象



def process_with_multiprocessing(img_floder_path,output_video,fps,threads):
    seqs = [(seq,out_seq,fps) for seq,out_seq in zip(img_floder_path,output_video)]
    with multiprocessing.Pool(threads) as pool:
        # 创建进度条
        with tqdm(total=len(seqs)) as pbar:
            # 定义一个回调函数，在每个任务完成时更新进度条
            def update(*a):
                pbar.update(1)
            # 使用 apply_async 处理并行任务，并将回调函数传入
            for seq in seqs:
                pool.apply_async(images_to_video, args=seq, callback=update)
            # 关闭池，等待所有任务完成
            pool.close()
            pool.join()

threads = 5
image_folder = ''  
save_path = ''
direct_video_name = ['car_005','car_007','car_028','car_large_032','car_large_007']
if not os.path.exists(save_path):
    os.makedirs(save_path)
img_floder_path = [os.path.join(image_folder,video) for video in sorted(os.listdir(image_folder)) if video in direct_video_name]
videos_name = [video for video in sorted(os.listdir(image_folder)) if video in direct_video_name]
output_video = [save_path + video + '.mp4' for video in videos_name]
fps = 30  
process_with_multiprocessing(img_floder_path,output_video,fps,threads)