import os
import cv2
import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_configs import RecordControlConfig
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config, make_robot
from lerobot.common.robot_devices.robots.configs import RobotConfig, StretchRobotConfig

import shutil
dataset_root = 'data/cuarm_small_9_12'
single_task = "Pick the cup and drop it in the table."
# os.removedirs(dataset_root)
shutil.rmtree(dataset_root, ignore_errors=True)

# episode_num = 1
# fps = 100.0
raw_fps = 100.0
data_fps = 20
# crop_paras = 240, 700, 420, 1260
crop_paras = None
# 25

def downsample_data(joint_data, raw_fps=100.0, data_fps=30.0):
    # 计算采样比例
    ratio = raw_fps / data_fps
    
    # 创建新的时间索引
    indices = np.arange(0, len(joint_data), ratio)  # 生成以 `ratio` 为步长的索引
    
    # 对索引进行四舍五入并转换为整数
    indices = np.round(indices).astype(int)
    
    # 防止索引超出范围
    indices = indices[indices < len(joint_data)]
    
    # 根据索引选取降采样后的数据
    downsampled_data = joint_data[indices]
    return downsampled_data


def get_frames_by_indices(video_path, indices):
    # 打开视频文件
    video_cap = cv2.VideoCapture(video_path)
    
    # 获取视频的帧数（用于验证索引是否有效）
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in the video: {total_frames}")
    
    frames = []
    
    for index in indices:
        if index < total_frames:
            # 设置视频帧的位置
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            
            # 读取该帧
            ret, frame = video_cap.read()
            if ret:
                frames.append(frame)  # 将帧添加到列表
            else:
                print(f"Failed to read frame at index {index}")
        else:
            print(f"Index {index} is out of bounds.")
    
    # 释放视频捕获对象
    video_cap.release()
    
    # 返回帧数组
    return np.array(frames)



cfg = RecordControlConfig(
    repo_id="cao63445538/cuarm_s",
    single_task=single_task,
    root=dataset_root,
    policy=None,
    fps=data_fps,
    warmup_time_s=0.1,
    episode_time_s=60,
    reset_time_s=60,
    num_episodes=50,
    video=True,
    push_to_hub=False,
    private=False,
)

robot = make_robot('cuarm_s')

dataset = LeRobotDataset.create(
    cfg.repo_id,
    cfg.fps,
    root=cfg.root,
    robot=robot,
    use_videos=cfg.video,
    image_writer_processes=cfg.num_image_writer_processes,
    image_writer_threads=cfg.num_image_writer_threads_per_camera,
)

raw_data_root = 'data/small_9_12'

for episode_num in range(0, 3):

    joint_data_path = os.path.join(raw_data_root, '{}'.format(episode_num), 'data.txt')
    main_video_data_path = os.path.join(raw_data_root, '{}'.format(episode_num), 'Kinect.avi')
    hand_video_data_path = os.path.join(raw_data_root, '{}'.format(episode_num), 'UsbCam.avi')
    
    if not os.path.exists(joint_data_path):
        print(f"Data files for episode {episode_num} are missing. Skipping this episode.")
        continue
    
    if not os.path.exists(main_video_data_path):
        single_arm = True
    else:
        single_arm = False
        
    joint_data = np.loadtxt(joint_data_path)
    vaild_data_mask = joint_data[:, -4] >= 0
    joint_data = joint_data[vaild_data_mask]

    sampled_joint_data = downsample_data(joint_data, raw_fps, data_fps)

    # TODO: modify the index position
    sampled_hand_frames = get_frames_by_indices(hand_video_data_path, sampled_joint_data[:, -4].astype(int))
    if not single_arm:
        sampled_main_frames = get_frames_by_indices(main_video_data_path, sampled_joint_data[:, -4].astype(int))
    
    if not single_arm:
        for data_sample, main_frame, hand_frame in zip(sampled_joint_data, sampled_main_frames, sampled_hand_frames):

            joint_state = np.append(data_sample[7:14], data_sample[-1:])
            observation, action = {}, {}
            observation["observation.state"] = torch.from_numpy(joint_state).float()
            action["action"] = torch.from_numpy(joint_state).float()

            if crop_paras[0] != 0 or crop_paras is not None:
                main_frame = main_frame[crop_paras[0]:crop_paras[1], crop_paras[2]:crop_paras[3], :]
            resize_main_frame = cv2.resize(main_frame, (640, 480))
            resize_main_frame = cv2.cvtColor(resize_main_frame, cv2.COLOR_BGR2RGB) / 255.0
            observation[f"observation.images.kinect"] = torch.from_numpy(resize_main_frame).float()

            hand_frame = cv2.rotate(hand_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            resize_hand_frame = cv2.resize(hand_frame, (640, 480))
            resize_hand_frame = cv2.cvtColor(resize_hand_frame, cv2.COLOR_BGR2RGB) / 255.0
            observation[f"observation.images.webcam"] = torch.from_numpy(resize_hand_frame).float()
            
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

    else:
        for data_sample, hand_frame in zip(sampled_joint_data, sampled_hand_frames):

            joint_state = np.append(data_sample[:6], data_sample[-2:-1])
            observation, action = {}, {}
            observation["observation.state"] = torch.from_numpy(joint_state).float()
            action["action"] = torch.from_numpy(joint_state).float()

            hand_frame = cv2.rotate(hand_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            resize_hand_frame = cv2.resize(hand_frame, (640, 480))
            resize_hand_frame = cv2.cvtColor(resize_hand_frame, cv2.COLOR_BGR2RGB) / 255.0
            observation[f"observation.images.webcam"] = torch.from_numpy(resize_hand_frame).float()
            
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)
    
    # dataset.clear_episode_buffer()
    dataset.save_episode()