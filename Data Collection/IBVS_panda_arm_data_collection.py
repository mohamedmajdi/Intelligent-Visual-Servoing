import numpy as np
import pybullet as p
import pybullet_data
import time
import math
#import cv2
#import csv
import os
from PIL import Image
import random
import json

# Connect to PyBullet GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.setTimeStep(0.01)
plane_id = p.loadURDF("plane.urdf")
wood_texture_id = p.loadTexture("wooden_texture.jpg")
p.changeVisualShape(plane_id, -1, textureUniqueId=wood_texture_id)

# Load Franka Panda
panda_model = "franka_panda/panda.urdf"
franka_id = p.loadURDF(panda_model, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

# Joint velocity limits
joint_velocity_limits = [150, 150, 150, 150, 180, 180, 180]
joint_velocity_limits = [v * (math.pi / 180) for v in joint_velocity_limits]

for i in range(len(joint_velocity_limits)):
    p.changeDynamics(bodyUniqueId=franka_id, linkIndex=i, maxJointVelocity=joint_velocity_limits[i])

# Camera intrinsics
fov, aspect, nearplane, farplane = 90, 1.0, 0.01, 100
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)

# define objects
radius = 0.035
visual_shape_id1 = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1])
collision_shape_id1 = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
ball_id1 = p.createMultiBody(baseMass=0,
                            baseCollisionShapeIndex=collision_shape_id1,
                            baseVisualShapeIndex=visual_shape_id1,
                            basePosition=[0.5, 0.5, 0])

visual_shape_id2 = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0, 0, 1, 1])
collision_shape_id2 = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
ball_id2 = p.createMultiBody(baseMass=0,
                            baseCollisionShapeIndex=collision_shape_id2,
                            baseVisualShapeIndex=visual_shape_id2,
                            basePosition=[0.5, 0.5, 0])

# Box1
box_col_id1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
box_vis_id1 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[0, 0, 1, 1])
box_id1 = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=box_col_id1,
                           baseVisualShapeIndex=box_vis_id1, basePosition=[0.7, 0.7, 0.1])

# Box2
box_col_id2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
box_vis_id2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[1,0,0, 1])
box_id2 = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=box_col_id2,
                           baseVisualShapeIndex=box_vis_id2, basePosition=[0.7, 0.7, 0.1])

# Get camera view matrix from end-effector 
def panda_camera():
    com_p, com_o, _, _, _, _ = p.getLinkState(franka_id, 11, computeForwardKinematics=True)
    rot_matrix = p.getMatrixFromQuaternion(com_o)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    camera_vector = rot_matrix.dot((0, 0, 1))
    up_vector = rot_matrix.dot((0, 1, 0))
    view_matrix = p.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
    img = p.getCameraImage(640, 480, view_matrix, projection_matrix)
    return img


controlled_joints = [i for i in range(p.getNumJoints(franka_id)) if p.getJointInfo(franka_id, i)[2] != p.JOINT_FIXED]
dataset_dir = "tiny_reach_dataset/reach_target_4_objects-testing"

metadata = []

objects = [
    (ball_id1, "reach the red ball", "red_ball"),
    (ball_id2, "reach the blue ball", "blue_ball"),
    (box_id1, "reach the blue box", "blue_box"),
    (box_id2, "reach the red box", "red_box"),
]


#collect demos for multiple objects at the same time
demo_counter = 0  # This will increment for every demo, regardless of object

for obj_id, prompt, obj_name in objects:
    for demo in range(63):  # Or however many demos per object you want
        demo_dir = os.path.join(dataset_dir, f"demo_{demo_counter}")
        os.makedirs(demo_dir, exist_ok=True)
        home_position = [0, -0.4, 0, -1.8, 0, 1.4, 0]
        
        # Move robot to home position
        for i in range(len(home_position)):
            p.setJointMotorControl2(franka_id, i, p.POSITION_CONTROL, targetPosition=home_position[i])
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)
        
        # Randomize ALL object positions
        target_pos_ball1 = [random.uniform(0.3, 0.7), random.uniform(-0.4, 0.4), 0.02]
        p.resetBasePositionAndOrientation(ball_id1, target_pos_ball1, [0, 0, 0, 1])
        target_pos_ball2 = [random.uniform(0.3, 0.7), random.uniform(-0.4, 0.4), 0.02]
        p.resetBasePositionAndOrientation(ball_id2, target_pos_ball2, [0, 0, 0, 1])
        target_pos_box1 = [random.uniform(0.3, 0.7), random.uniform(-0.5, 0.5), 0]
        p.resetBasePositionAndOrientation(box_id1, target_pos_box1, [0, 0, 0, 1])
        target_pos_box2 = [random.uniform(0.3, 0.7), random.uniform(-0.5, 0.5), 0]
        p.resetBasePositionAndOrientation(box_id2, target_pos_box2, [0, 0, 0, 1])
        
        # Choose correct target position for this object
        if obj_id == ball_id1:
            target_pos = target_pos_ball1
        elif obj_id == ball_id2:
            target_pos = target_pos_ball2
        elif obj_id == box_id1:
            target_pos = target_pos_box1
        elif obj_id == box_id2:
            target_pos = target_pos_box2
        
        # Collect 5 frames
        for step in range(5):
            img = panda_camera()
            rgb = np.reshape(img[2], (480, 640, 4))[:, :, :3]
            img_pil = Image.fromarray(rgb).resize((256, 256))
            img_pil.save(os.path.join(demo_dir, f"frame_{step:02d}.png"))
            
            # Move joints toward current object's target
            joint_positions = p.calculateInverseKinematics(
                franka_id, 11, target_pos, maxNumIterations=100
            )
            for joint_id, joint_pos in zip(controlled_joints, joint_positions):
                p.setJointMotorControl2(
                    franka_id, joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=joint_pos,
                    force=500
                )
            for _ in range(8):
                p.stepSimulation()
                time.sleep(0.01)
        
        # Metadata for this demo
        for step in range(4):
            metadata.append({
                "original_image": f"demo_{demo_counter}/frame_{step:02d}.png",
                "edited_image": f"demo_{demo_counter}/frame_{step+1:02d}.png",
                "edit_prompt": prompt,
                "file_name": f"demo_{demo_counter}/frame_{step:02d}.png",
            })
        
        demo_counter += 1  # Move to next demo number

p.disconnect()




'''
# Collect demonstrations for one object only
for demo in range(5):
    demo_dir = os.path.join(dataset_dir, f"demo_{demo}")
    os.makedirs(demo_dir, exist_ok=True)
    home_position = [0, -0.4, 0, -1.8, 0, 1.4, 0] # Set to a neutral/home configuration
    

    # Move the robot to the home position
    for i in range(len(home_position)):
        p.setJointMotorControl2(franka_id, i, p.POSITION_CONTROL, targetPosition=home_position[i])

    # Simulate for a few steps to let the robot settle into the home position before running the main simulation loop
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.01)
   
    # Random positions
    target_pos_ball1 = [random.uniform(0.3, 0.7), random.uniform(-0.4, 0.4), 0.02]
    p.resetBasePositionAndOrientation(ball_id1, target_pos_ball1, [0, 0, 0, 1])
    
    target_pos_ball2 = [random.uniform(0.3, 0.7), random.uniform(-0.4, 0.4), 0.02]
    p.resetBasePositionAndOrientation(ball_id2, target_pos_ball2, [0, 0, 0, 1])
    
    target_pos_box1 = [random.uniform(0.3, 0.7), random.uniform(-0.5, 0.5), 0]
    p.resetBasePositionAndOrientation(box_id1, target_pos_box1, [0, 0, 0, 1])
    
    target_pos_box2 = [random.uniform(0.3, 0.7), random.uniform(-0.5, 0.5), 0]
    p.resetBasePositionAndOrientation(box_id2, target_pos_box2, [0, 0, 0, 1])
    
        
    # Collect 5 frames
    for step in range(5):
        # Get and save image
        img = panda_camera()
        rgb = np.reshape(img[2], (480, 640, 4))[:, :, :3]
        img_pil = Image.fromarray(rgb).resize((256, 256))
        img_pil.save(os.path.join(demo_dir, f"frame_{step:02d}.png"))
        
        # Move joints toward target
        joint_positions = p.calculateInverseKinematics(
            franka_id, 11, target_pos_ball1,
            maxNumIterations=100
        )
        for joint_id, joint_pos in zip(controlled_joints, joint_positions):
            p.setJointMotorControl2(
                franka_id, joint_id,
                p.POSITION_CONTROL,
                targetPosition=joint_pos,
                force=500
            )
    
        # Simulate a bit longer to allow visible motion
        for _ in range(8):  # simulate 8 steps 
            p.stepSimulation()
            time.sleep(0.01)  # optional: slow down for real-time viewing
    
    # Create metadata entries for pairs
    for step in range(4):  # Create 4 pairs from 5 frames
        metadata.append({
            "original_image": f"demo_{demo}/frame_{step:02d}.png",
            "edited_image": f"demo_{demo}/frame_{step+1:02d}.png",
            "edit_prompt": "reach the red ball",
            "file_name": f"demo_{demo}/frame_{step:02d}.png"
        })


p.disconnect()
'''
'''
# Create prompts file
with open(os.path.join(dataset_dir, "prompts.csv"), "w") as f:
    f.write("input_path,target_path,prompt\n")
    for demo in range(250):
        for step in range(4):  # 4 pairs from 5 frames
            f.write(
                f"./demo_{demo}/frame_{step:02d}.png,"
                f"./demo_{demo}/frame_{step+1:02d}.png,"
                "\"reach the red ball\"\n"
            )
'''
# Write metadata.jsonl file
metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
with open(metadata_path, "w") as f:
    for entry in metadata:
        f.write(json.dumps(entry) + "\n")
        
print("Dataset generated with PyBullet.")