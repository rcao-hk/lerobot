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
raw_fps = 50.0
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

# def downsample_data(joint_data, raw_fps=100.0, data_fps=30.0):
#     """
#     按时间轴选择采样点，避免 round 漂移与重复索引。
#     """
#     n = len(joint_data)
#     if n == 0:
#         return joint_data
#     # 原始每行对应的时间戳
#     t_raw = np.arange(n, dtype=np.float64) / float(raw_fps)
#     t_end = t_raw[-1]
#     # 目标时间轴：从 0 到 t_end（含）
#     t_tar = np.arange(0.0, t_end + 1e-9, 1.0 / float(data_fps))
#     # 找到最接近的原始索引（向左）
#     idx = np.searchsorted(t_raw, t_tar, side="left")
#     idx[idx >= n] = n - 1
#     # 去重，严格单调不降
#     idx = np.unique(idx)
#     return joint_data[idx]


# def get_frames_by_indices(video_path, indices):
#     # 打开视频文件
#     video_cap = cv2.VideoCapture(video_path)
    
#     # 获取视频的帧数（用于验证索引是否有效）
#     total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"Total frames in the video: {total_frames}")
    
#     frames = []
    
#     for index in indices:
#         if index < total_frames:
#             # 设置视频帧的位置
#             video_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            
#             # 读取该帧
#             ret, frame = video_cap.read()
#             if ret:
#                 frames.append(frame)  # 将帧添加到列表
#             else:
#                 print(f"Failed to read frame at index {index}")
#         else:
#             print(f"Index {index} is out of bounds.")
    
#     # 释放视频捕获对象
#     video_cap.release()
    
#     # 返回帧数组
#     return np.array(frames)


def get_frames_by_indices(video_path, indices):
    """
    顺序读取整段视频，但严格按传入 indices 的“原始顺序 + 重复”返回帧。
    若某些帧缺失，用上一帧兜底，保证返回长度与 indices 相同。
    """
    if not os.path.exists(video_path):
        print(f"[vid] MISSING: {video_path}")
        return np.empty((0,), dtype=np.uint8)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[vid] ERROR: cannot open {video_path}")
        return np.empty((0,), dtype=np.uint8)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 1) 原样保留顺序和重复
    idx_orig = np.asarray(indices, dtype=int)
    idx_orig = idx_orig[idx_orig >= 0]
    if total > 0:
        idx_orig = np.clip(idx_orig, 0, total - 1)
    if len(idx_orig) == 0:
        cap.release()
        return np.empty((0,), dtype=np.uint8)

    # 2) 建立 “帧号 -> 输出位置列表”
    wanted_positions = {}
    for out_pos, fno in enumerate(idx_orig):
        fno = int(fno)
        wanted_positions.setdefault(fno, []).append(out_pos)

    # 3) 顺序扫描视频，命中帧就填到对应输出位置（可能有多个重复）
    outputs = [None] * len(idx_orig)
    last = None
    fno = 0
    ok, frame = cap.read()
    while ok:
        if fno in wanted_positions:
            for out_pos in wanted_positions[fno]:
                outputs[out_pos] = frame.copy()
            last = frame
            if all(x is not None for x in outputs):
                break
        fno += 1
        ok, frame = cap.read()
    cap.release()

    # 4) 兜底：未填的位置用上一帧补
    for i in range(len(outputs)):
        if outputs[i] is None:
            if last is not None:
                outputs[i] = last.copy()
            else:
                return np.empty((0,), dtype=np.uint8)

    return np.stack(outputs, axis=0)  # uint8, N x H x W x 3


def load_cam_sidecar(sidecar_path):
    """
    读取 camera 侧的时间戳 sidecar（由采集脚本写的 UsbCam_timestamps.txt）
    返回 (frame_idx_arr[int], mono_ns_arr[int])；不存在则返回 (None, None)
    文件格式（tab 分隔）：
    frame_idx    mono_ns            wall_ns
    0            6523...            1726...
    """
    if not os.path.exists(sidecar_path):
        return None, None
    try:
        # 跳过第一行表头，tab 分隔
        data = np.loadtxt(sidecar_path, dtype=np.int64, delimiter='\t', skiprows=1)
        # 兼容只有一行时的 shape
        if data.ndim == 1:
            data = data[None, :]
        fids = data[:, 0].astype(np.int64)
        monos = data[:, 1].astype(np.int64)
        return fids, monos
    except Exception as e:
        print(f"[sidecar] WARN: failed to read {sidecar_path}: {e}")
        return None, None


def map_mono_to_frameidx(side_mono_ns, side_frame_idx, target_mono_ns):
    """
    用“最近邻”把 target_mono_ns（N,）映射到 sidecar 的 frame_idx。
    side_mono_ns 必须单调不降；若为空，返回 None。
    """
    if side_mono_ns is None or side_frame_idx is None or len(side_mono_ns) == 0:
        return None
    target = np.asarray(target_mono_ns, dtype=np.int64)

    # 二分找右邻
    pos = np.searchsorted(side_mono_ns, target, side='left')
    pos = np.clip(pos, 0, len(side_mono_ns) - 1)
    left = np.clip(pos - 1, 0, len(side_mono_ns) - 1)

    # 选择更近者
    choose_left = np.abs(side_mono_ns[left] - target) <= np.abs(side_mono_ns[pos] - target)
    best = np.where(choose_left, left, pos)

    return side_frame_idx[best].astype(int)


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

for episode_num in range(0, 20):

    joint_data_path = os.path.join(raw_data_root, '{}'.format(episode_num), 'data.txt')
    main_video_data_path = os.path.join(raw_data_root, '{}'.format(episode_num), 'Kinect.avi')
    hand_video_data_path = os.path.join(raw_data_root, '{}'.format(episode_num), 'UsbCam.avi')
    
    if not os.path.exists(joint_data_path):
        print(f"Data files for episode {episode_num} are missing. Skipping this episode.")
        continue

    joint_data = np.loadtxt(joint_data_path)

    if not os.path.exists(main_video_data_path):
        single_arm = True
        vaild_data_mask = joint_data[:, -3] >= 0   # 没主相机就用手摄的可用标记
    else:
        vaild_data_mask = joint_data[:, -4] >= 0
        single_arm = False

    joint_data = joint_data[vaild_data_mask]
    sampled_joint_data = downsample_data(joint_data, raw_fps, data_fps)

    # ==== 时间戳对齐：读取 sidecar ====
    hand_sidecar = os.path.join(raw_data_root, str(episode_num), "UsbCam_timestamps.txt")
    hand_fids_side, hand_monos_side = load_cam_sidecar(hand_sidecar)

    # === 从 data.txt 中取时间戳列 ===
    # 末尾 4 列（如果按我之前建议的采集格式）：
    # -4: ctrl_mono_ns, -3: cam_fid_dup, -2: cam_mono_ns, -1: ctrl_wall_ns
    # 如果你的 data.txt 还没有这些列，就把 cam_mono_ns_col 设为 None，会自动回退到旧的 image_id 列。
    cam_mono_ns_col = None
    if sampled_joint_data.shape[1] >= 32:  # 粗略判断是否包含新增的 4 列
        cam_mono_ns_col = sampled_joint_data[:, -2].astype(np.int64)  # cam_mono_ns
        # 也可以选用 ctrl_mono_ns 做匹配：
        # ctrl_mono_ns_col = sampled_joint_data[:, -4].astype(np.int64)

    # === 用时间戳映射到帧号；失败则回退到旧列 ===
    if cam_mono_ns_col is not None and hand_monos_side is not None:
        hand_indices = map_mono_to_frameidx(hand_monos_side, hand_fids_side, cam_mono_ns_col)
    else:
        # 回退：沿用旧的 image_id 列（你的 -3 是手摄，-4 是主相机）
        hand_indices = sampled_joint_data[:, -3].astype(int)

    # 取手摄帧
    sampled_hand_frames = get_frames_by_indices(hand_video_data_path, hand_indices)

    if not single_arm:
        # 若有主相机，也读取它的 sidecar（命名自行对应）
        main_sidecar = os.path.join(raw_data_root, str(episode_num), "Kinect_timestamps.txt")
        main_fids_side, main_monos_side = load_cam_sidecar(main_sidecar)

        # 主相机的时间戳列（若你也在 data.txt 里写了主相机的 mono_ns，替换这里）
        # 默认回退到旧列 -4：
        main_indices = None
        if main_monos_side is not None and cam_mono_ns_col is not None:
            # 若你只有一份相机 mono_ns（手摄），主相机暂时用旧列
            main_indices = sampled_joint_data[:, -4].astype(int)
        else:
            main_indices = sampled_joint_data[:, -4].astype(int)

        sampled_main_frames = get_frames_by_indices(main_video_data_path, main_indices)

    # ---- 对齐长度，避免 zip 早停或越界 ----
    if not single_arm:
        min_len = min(len(sampled_joint_data), len(sampled_main_frames), len(sampled_hand_frames))
    else:
        min_len = min(len(sampled_joint_data), len(sampled_hand_frames))

    sampled_joint_data = sampled_joint_data[:min_len]
    sampled_hand_frames = sampled_hand_frames[:min_len]
    if not single_arm:
        sampled_main_frames = sampled_main_frames[:min_len]
        
    if not single_arm:
        for data_sample, main_frame, hand_frame in zip(sampled_joint_data, sampled_main_frames, sampled_hand_frames):

            joint_state = np.append(data_sample[7:14], data_sample[-1:])
            observation, action = {}, {}
            observation["observation.state"] = torch.from_numpy(joint_state).float()
            action["action"] = torch.from_numpy(joint_state).float()

            # 裁剪判断：先判断是否 None
            if crop_paras is not None:
                y1, y2, x1, x2 = crop_paras
                main_frame = main_frame[y1:y2, x1:x2, :]
                
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

            joint_state = np.append(data_sample[:6], data_sample[26:27])
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