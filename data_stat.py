import numpy as np



root = 'data/7_17'
root = 'data/8_4'
heights = []
for episode_num in range(1, 21):
    joint_data = np.loadtxt(f'{root}/{episode_num}/data.txt')
    
    vaild_data_mask = joint_data[:, -4] >= 0
    height = joint_data[:, -8]
    heights.append(np.mean(height[vaild_data_mask]))
print(np.mean(heights))
