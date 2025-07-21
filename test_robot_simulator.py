import sys
import signal
from functools import partial
import time
from udp_socket import UdpSocket
import random

def signal_handler(udp, sig, frame):
    print('机器人模拟器收到Ctrl+C信号!')
    udp.close()
    sys.exit(0)

if __name__ == '__main__':
    # 机器人端配置（与inference.py中的remote和local相反）
    local_ip, local_port = "127.0.0.1", 10088
    remote_ip, remote_port = "127.0.0.1", 10087
    
    udp = UdpSocket(local_ip, local_port, remote_ip, remote_port)
    signal.signal(signal.SIGINT, partial(signal_handler, udp))
    udp.open()
    
    print(f"机器人模拟器启动: {local_ip}:{local_port} -> {remote_ip}:{remote_port}")
    
    for i in range(1000):
        # 模拟发送observation数据
        # 格式: "kinect_image_data#webcam_image_data#state_values"
        
        # 生成8个状态值
        state_values = [random.uniform(-1, 1) for _ in range(8)]
        state_str = ','.join([f"{x:.6f}" for x in state_values])
        
        # 目前使用占位符表示图像数据
        # TODO: 后续可以替换为实际的图像字节数据
        kinect_data = "kinect_placeholder"  # 实际应该是编码后的图像数据
        webcam_data = "webcam_placeholder"  # 实际应该是编码后的图像数据
        
        observation_data = f"{kinect_data}#{webcam_data}#{state_str}"
        udp.send(observation_data)
        print(f"[{i+1}] 发送observation:")
        print(f"  - kinect: {kinect_data}")
        print(f"  - webcam: {webcam_data}")  
        print(f"  - state: {state_str}")
        
        # 接收action结果
        action_data = udp.receive(dt=0.02)
        if action_data:
            action_values = action_data.split(',')
            print(f"[{i+1}] ✓ 接收action: {len(action_values)}个值")
            print(f"  前3个值: {action_values[:3] if len(action_values) >= 3 else action_values}")
        else:
            print(f"[{i+1}] ✗ 未接收到action")
        
        print("-" * 50)
        time.sleep(0.05)  # 20Hz
    
    udp.close()
