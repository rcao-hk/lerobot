from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.utils import busy_wait
import time
import torch

import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
print(torch.profiler.supported_activities())   # 只会输出 {ProfilerActivity.CPU}

inference_time_s = 120
fps = 20
device = "cuda"  # TODO: On Mac, use "mps" or "cpu"

ckpt_path = "outputs/train/cuarm/checkpoints/003000/pretrained_model"
policy = ACTPolicy.from_pretrained(ckpt_path)
policy.to(device)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    # observation = robot.capture_observation()
    
    observation = {}
    observation['observation.images.kinect'] = torch.rand((1, 3, 480, 640), device=device)
    observation['observation.images.webcam'] = torch.rand((1, 3, 480, 640), device=device)
    observation['observation.state'] = torch.rand((1, 8), device=device)
    
    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    # for name in observation:
    #     observation[name] = observation[name].type(torch.float32) / 255
    #     observation[name] = observation[name].permute(2, 0, 1).contiguous()
    #     observation[name] = observation[name].unsqueeze(0)
    #     observation[name] = observation[name].to(device)

    # Compute the next action with the policy
    # based on the current observation
    start.record()
    action = policy.select_action(observation)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    # action = action.to("cpu")
    # Order the robot to move
    
    # robot.send_action(action)
    # dt_s = time.perf_counter() - start_time
    # print(f"FPS: {1 / dt_s:.2f} | Time: {dt_s:.4f}s")

    end.record()
    torch.cuda.synchronize()
    dt_s = start.elapsed_time(end) / 1000.0
    print(f"FPS: {1 / dt_s:.2f} | Time: {dt_s:.4f}s")
      
    # busy_wait(1 / fps - dt_s)