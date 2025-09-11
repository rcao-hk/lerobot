import numpy as np
import matplotlib.pyplot as plt


# fps 1/3500

def plot_joints_and_states(file_path):
    # Load the data from the txt file
    data = np.loadtxt(file_path)
    
    # Extract the left_joint_pos (columns 0 to 6) and right_joint_pos (columns 7 to 13)
    left_joint_pos = data[:, 0:7]
    right_joint_pos = data[:, 7:14]
    # left_joint_pos = data[:, 14:20]
    # right_joint_pos = data[:, 20:26]
    
    # Extract the camera_wrote_frame (columns 13 and 14)
    camera_wrote_frame = data[:, -4:-2]
    
    # Extract the gripper_state (columns 15 and 16)
    gripper_state = data[:, -2:] # 0 - open, 1 - close
    
    # Plot the joint data (j1 - j6) in one plot
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(right_joint_pos[:, i], label=f'Joint {i+1}')
    plt.plot(camera_wrote_frame[:, 0]/100, label='kinect')
    # plt.plot(camera_wrote_frame[:, 1], label='webcam')
    plt.plot(gripper_state[:, 0], label='Gripper State 1')
    plt.title('Joint Position (j1 - j6)')
    plt.xlabel('Frame')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('left_joint.png')
    
    # Plot the camera_wrote_frame and gripper_state in another plot
    plt.figure(figsize=(10, 6))
    plt.plot(camera_wrote_frame[:, 0], label='kinect')
    # plt.plot(camera_wrote_frame[:, 1], label='webcam')
    plt.plot(gripper_state[:, 0], label='Gripper State 1')
    # plt.plot(gripper_state[:, 1], label='Gripper State 2')
    plt.title('Camera Wrote Frame & Gripper State')
    plt.xlabel('Frame')
    plt.ylabel('State')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('left_camera.png')

# Example usage:
# Assuming the file path is 'data.txt', you can call the function as:
plot_joints_and_states('data/7_17/1/data.txt')

