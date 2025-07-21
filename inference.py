import cv2
cv2.namedWindow("kinect_frame", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("webcam_frame", cv2.WINDOW_AUTOSIZE)

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.utils import busy_wait
import time
import torch
import sys
import signal
from functools import partial
from udp_socket import UdpSocket
import json
import numpy as np
import threading
import sys
from pathlib import Path

#Add kinect 
parent_path =str(Path(__file__).parent.absolute().parent.absolute())
cameras_path = str(parent_path) + "/depth_sensor_new"
sys.path.extend([parent_path, cameras_path])
import scripts.utils as utils
from kinect.python.scripts.kinect import Kinect

#Add udp 
configuration_path = str(parent_path) + "/cuarm_configuration/share/python"
sys.path.extend([configuration_path])
from cuarm_configuration.share.python.cuarm_udp import CuarmUdpThread, CuarmUdp
from cuarm_configuration.share.python.cuarm_state import CuroboState, TargetCommand, TargetCommandMode, GripperMode


print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

fps = 10
camera_fps = 10
recv_robot_state_ms = 0.01
device = "cuda"  # TODO: On Mac, use "mps" or "cpu"

ckpt_path = "outputs/cuarm/checkpoints/003000/pretrained_model"
policy = ACTPolicy.from_pretrained(ckpt_path)
policy.to(device)

# 设置UDP通信
local_ip, local_port = "127.0.0.1", 10091  # 推理端的IP和端口
remote_ip, remote_port = "127.0.0.1", 10092  # 机器人端的IP和端口
udp = CuarmUdpThread(local_ip, local_port, remote_ip, remote_port, 
                     unpack_message_func=CuarmUdp.unpack_curobo_state, pack_message_func=CuarmUdp.pack_target_command, recv_delay_ms=recv_robot_state_ms)
print(f"UDP连接已建立: 本地 {local_ip}:{local_port} -> 远程 {remote_ip}:{remote_port}")
shut_down = False

file = open("data.txt", "w") 

def signal_handler(udp, sig, frame):
    print('接收到Ctrl+C信号，正在关闭UDP连接...')
    global shut_down
    shut_down = True
    udp.terminate()
    kinect_thread.join()
    webcam_thread.join()
    file.close()
    sys.exit(0)

signal.signal(signal.SIGINT, partial(signal_handler, udp))
# udp.open()

frame_lock = threading.Lock()
frame_dimensions = (640, 480) #width, height
kinect_frame = np.zeros((*frame_dimensions, 3), dtype=np.uint8)
webcam_frame = np.zeros((*frame_dimensions, 3), dtype=np.uint8)

def kinect_task():
    kinect_cam = Kinect()
    start_time = time.perf_counter()
    while not shut_down:
        kinect_cam.update()
        frame = kinect_cam.get_color_image()
        if frame is not None:
            frame = cv2.resize(frame, frame_dimensions, interpolation = cv2.INTER_AREA)
            with frame_lock:
                global kinect_frame
                kinect_frame = frame.copy()
        
        total_time = time.perf_counter() - start_time
        # print("kinect total_time: ", total_time , "s")
        busy_wait(1 / camera_fps - total_time)
        start_time = time.perf_counter()

def webcam_task():
    webcam = cv2.VideoCapture(0)
    start_time = time.perf_counter()
    while not shut_down:
        ret, frame = webcam.read()
        if frame is not None:
            with frame_lock:
                global webcam_frame
                webcam_frame = frame.copy()
        
        total_time = time.perf_counter() - start_time
        # print("webcam total_time: ", total_time , "s")
        busy_wait(1 / camera_fps - total_time)
        start_time = time.perf_counter()

def parse_observation_data(state:CuroboState , kinect_frame, webcam_frame):
    """解析从机器人端接收的数据字符串为observation字典"""
    observation = {}
    try:
        # 解析state数据 (8个浮点数，用逗号分隔)
        try:
            right_arm_joint_position = state.arm_joint_position[7:]
            right_hand_position = [int(prev_action_gripper_cmd.value)]
            state_values = np.concatenate([right_arm_joint_position, right_hand_position])
            print("state_values:", state_values)
            if len(state_values) == 8:
                observation['observation.state'] = torch.tensor(state_values, device=device).float().unsqueeze(0)
                # print(f"✓ 状态数据解析成功: shape {observation['observation.state'].shape}")
            else:
                print(f"状态数据长度错误: 期望8个值，实际{len(state_values)}个")
                return None
        except ValueError as e:
            print(f"状态数据解析失败: {e}")
            return None
        
        # 解析kinect图像数据
        # 目前使用模拟数据，保持正确的shape: (1, 3, 480, 640)
        # TODO: 后续可以解析实际的图像字节数据
        if kinect_frame is not None:
            kinect_data = np.expand_dims(kinect_frame.transpose(2, 0, 1), axis=0)
            observation['observation.images.kinect'] =  torch.tensor(kinect_data, device=device)
            # print("✓ Kinect图像数据已生成: shape (1, 3, 480, 640)")
        else:
            # 这里可以添加实际图像解析逻辑
            print("⚠ 暂不支持实际kinect图像解析，使用随机数据")
            observation['observation.images.kinect'] = torch.rand((1, 3, 480, 640), device=device)
        
        # 解析webcam图像数据  
        # 目前使用模拟数据，保持正确的shape: (1, 3, 480, 640)
        # TODO: 后续可以解析实际的图像字节数据
        if webcam_frame is not None:
            webcam_frame = cv2.rotate(webcam_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 旋转90度
            webcam_data = np.expand_dims(webcam_frame.transpose(2, 0, 1), axis=0)
            observation['observation.images.webcam'] = torch.tensor(webcam_data, device=device)
            # print("✓ Webcam图像数据已生成: shape (1, 3, 480, 640)")
        else:
            # 这里可以添加实际图像解析逻辑
            print("⚠ 暂不支持实际webcam图像解析，使用随机数据")
            observation['observation.images.webcam'] = torch.rand((1, 3, 480, 640), device=device)
        
        # 验证observation数据完整性
        expected_keys = ['observation.images.kinect', 'observation.images.webcam', 'observation.state']
        if all(key in observation for key in expected_keys):
            # print("✓ Observation数据完整，包含所有必需字段")
            # print(f"  - kinect: {observation['observation.images.kinect'].shape}")
            # print(f"  - webcam: {observation['observation.images.webcam'].shape}") 
            # print(f"  - state: {observation['observation.state'].shape}")
            return observation
        else:
            missing_keys = [key for key in expected_keys if key not in observation]
            print(f"✗ Observation数据不完整，缺少字段: {missing_keys}")
            return None
        
    except Exception as e:
        print(f"解析observation数据时出错: {e}")
        return None

def format_action_data(action, data):
    """将action tensor格式化为字符串以便UDP发送"""
    res = None
    try:
        # 将action转换为numpy数组再转为列表
        action_list = action.cpu().numpy()

        #Training data's J6 & J7 motor direction is reversed. So we need to reverse the direction of J6 & J7 motor.
        #TODO: Remove it for next training data
        # action_list[5], action_list[6] = -action_list[5], -action_list[6]


        if np.isnan(action_list).any():
            raise ValueError("Action contains NaN values.")
    
        left_arm_fixed_joint_position = np.array([55.070700, 56.936422, -25.906014, 47.423157, 29.365563, 34.855786, -30.930951])*np.pi/180

        cmd  = TargetCommand()
        cmd.mode = TargetCommandMode.JOINT_POSITION.value
        cmd.arm_joint_size = 14
        cmd.arm_joint_position = np.concatenate((left_arm_fixed_joint_position, action_list[:7]))

        cmd.gripper_size = 2
        cmd.gripper_mode_command = np.zeros(cmd.gripper_size, dtype=np.int32)
        if (action_list[-1] <= 0.5): #Open
            cmd.gripper_mode_command[1] = int(GripperMode.OPEN.value)  #Right hand only
        res = cmd
        # print("action_list_before: ", action_list_before)
        print("Send Joint Target: ", cmd.arm_joint_position)
    except Exception as e:
        print(f"格式化action数据时出错: {e}")
    return res, prev_action_joint_cmd, prev_action_gripper_cmd

def write_to_file(curobo_state, target_command):
    s = ""
    if curobo_state:
        s += ' '.join(map(str, curobo_state.arm_joint_position[7:])) + ' '
    if target_command and target_command.arm_joint_size == 14 and target_command.gripper_size == 2:
        s += ' '.join(map(str, target_command.arm_joint_position[7:])) + ' '
        s += str(target_command.gripper_mode_command[1]) + ' '
    file.write(s+"\n")

kinect_thread = threading.Thread(target=kinect_task)
webcam_thread = threading.Thread(target=webcam_task)
kinect_thread.start()
webcam_thread.start()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

success_count = 0
failed_count = 0

prev_arm_joint_position = np.zeros(14)
prev_action_joint_cmd = np.zeros(14)
prev_action_gripper_cmd = GripperMode.OPEN #Initialize gripper mode as open

data = None
cmd = TargetCommand()

start_time = time.perf_counter()
while True:
    if shut_down:
        break
    
    # print(f"\n=== 迭代 {iteration + 1} ===")
    
    # 接收机器人端的observation数据
    # print("等待接收机器人数据...")
    # data = udp.receive(dt=0.01)  # 等待10ms
    with frame_lock:
        kinect_frame_copy = kinect_frame.copy()
        webcam_frame_copy = webcam_frame.copy()
        
    cv2.imshow("kinect_frame", kinect_frame_copy)
    cv2.imshow("webcam_frame", webcam_frame_copy)
    cv2.waitKey(1)
    
    data: CuroboState = udp.receive()

    if data is not None and data.arm_joint_size == 14:
        # print(f"✓ 接收到数据: {data[:100]}..." if len(data) > 100 else f"✓ 接收到数据: {data}")
        
        # 解析数据为observation
        observation = parse_observation_data(data, kinect_frame_copy, webcam_frame_copy)
        
        if observation is None:
            print("✗ 数据解析失败，使用默认observation格式")
            # 使用正确格式的默认数据
            observation = {}
            observation['observation.images.kinect'] = torch.rand((1, 3, 480, 640), device=device)
            observation['observation.images.webcam'] = torch.rand((1, 3, 480, 640), device=device)
            observation['observation.state'] = torch.rand((1, 8), device=device)
            failed_count += 1
        else:
            success_count += 1
                
        # else:
        #     print("✗ 未接收到数据，使用默认observation格式")
        #     # 使用正确格式的默认数据，完全匹配原始格式
        #     observation = {}
        #     observation['observation.images.kinect'] = torch.rand((1, 3, 480, 640), device=device)
        #     observation['observation.images.webcam'] = torch.rand((1, 3, 480, 640), device=device)
        #     observation['observation.state'] = torch.rand((1, 8), device=device)
        #     failed_count += 1

        # 推理计算
        start.record()
        action = policy.select_action(observation)
        action = action.squeeze(0)
        end.record()
        torch.cuda.synchronize()
        
        inference_time = start.elapsed_time(end) / 1000.0
        # print(f"✓ 推理完成: Action shape {action.shape}, 用时 {inference_time:.4f}s")
        
        # 发送action结果
        cmd, prev_action_joint_cmd, prev_action_gripper_cmd = format_action_data(action, data)
        # if cmd:
        #     # udp.send(action_str)
        #     # print(f"✓ Action已发送: {action_str[:50]}..." if len(action_str) > 50 else f"✓ Action已发送: {action_str}")
        # else:
        #     print("✗ Action发送失败")
        #     failed_count += 1

    udp.send(cmd)
    write_to_file(data, cmd)


    # 计算总用时和FPS
    total_time = time.perf_counter() - start_time
    fps_actual = 1 / total_time if total_time > 0 else 0
    # print(f"总用时: {total_time:.4f}s | 实际FPS: {fps_actual:.2f} | 目标FPS: {fps}")
    # print(f"成功: {success_count}, 失败: {failed_count}, 成功率: {success_count/(success_count+failed_count)*100:.1f}%")
    
    # 控制循环频率
    busy_wait(1 / fps - total_time)
    end_time = time.perf_counter()
    start_time = end_time

# 关闭UDP连接
kinect_thread.join()
webcam_thread.join()
udp.terminate()
file.close()
# print(f"\n推理完成！总成功次数: {success_count}, 总失败次数: {failed_count}")