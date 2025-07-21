import torch
import torchvision.models as models
import time

# 设置设备为 CUDA（GPU），如果没有 CUDA 支持，自动切换为 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练的 ResNet-18 模型
model = models.resnet18(pretrained=True).to(device)
model.eval()  # 设置为评估模式，不进行训练

# 创建一个随机输入图像，尺寸为 (1, 3, 224, 224)，模拟一张 RGB 图像
# 注意：ResNet-18 输入图像要求大小为 224x224，且具有 3 个颜色通道（RGB）
input_tensor = torch.randn(1, 3, 480, 640).to(device)

# 进行一次预热推断，避免第一次推断由于初始化过程而时间较长
with torch.no_grad():
    model(input_tensor)

# 测试推断速度
num_iterations = 100  # 设置测试的推断次数
total_time = 0.0

# 计时并进行推断
for i in range(num_iterations):
    start_time = time.perf_counter()  # 记录开始时间

    with torch.no_grad():  # 不需要计算梯度
        output = model(input_tensor)  # 进行推断

    end_time = time.perf_counter()  # 记录结束时间
    total_time += (end_time - start_time)  # 累积推断时间

# 计算平均推断时间和 FPS（每秒推断次数）
average_inference_time = total_time / num_iterations
fps = num_iterations / total_time

print(f"Average inference time: {average_inference_time:.6f} seconds")
print(f"FPS: {fps:.2f}")
