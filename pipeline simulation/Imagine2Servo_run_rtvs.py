import numpy as np
import pybullet as p
import pybullet_data
import time
import math
import cv2
import csv
import random
from PIL import Image
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import matplotlib.pyplot as plt
import warnings
import os
import sys

from photo_error import mse_
from rtvs import Rtvs
import subprocess
import json
import base64
np.random.seed(0)

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode()

def call_rtvs(img_goal, img_src, pre_img_src):
    payload = {
        "img_goal": encode_image(img_goal),
        "img_src": encode_image(img_src),
        "pre_img_src": encode_image(pre_img_src)
    }

    conda_env_python = "/home/mohamed/miniconda3/envs/neuflow/bin/python"  # Change to your target env's Python
    proc = subprocess.Popen(
        [conda_env_python, "Running_rtvs.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    stdout, stderr = proc.communicate(input=json.dumps(payload).encode())
    if stderr:
        print("Error from subprocess:", stderr.decode())

    result = json.loads(stdout.decode())
    return result["vel"]







# the fine tuned diffusion model
model_id = "shahidhasib586/instruct-pix2pix-model" # <- the fine-tuned model in cluster https://huggingface.co/shahidhasib586/instruct-pix2pix-model
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)


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

# Add red ball
radius = 0.035
visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1])
collision_shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
ball_id = p.createMultiBody(baseMass=0,
                            baseCollisionShapeIndex=collision_shape_id,
                            baseVisualShapeIndex=visual_shape_id,
                            basePosition=[0.5, 0.5, 0])

'''# Box1
box_col_id1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
box_vis_id1 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[0, 0, 1, 1])
box_id1 = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=box_col_id1,
                           baseVisualShapeIndex=box_vis_id1, basePosition=[0.7, 0.7, 0.1])

# Box2
box_col_id2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
box_vis_id2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[0, 1, 1, 1])
box_id2 = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=box_col_id2,
                           baseVisualShapeIndex=box_vis_id2, basePosition=[0.7, 0.7, 0.1])
'''

'''# Ball sliders
ball_sliders = {
    'x': p.addUserDebugParameter('Ball X', -2, 2, 1.5),  #initial position of 1.5 along x-axis
    'y': p.addUserDebugParameter('Ball Y', -2, 2, 0.0),
    'z': p.addUserDebugParameter('Ball Z', 0, 1, 0)
}
def update_ball_position():
    pos = [p.readUserDebugParameter(ball_sliders[axis]) for axis in ['x', 'y', 'z']]
    p.resetBasePositionAndOrientation(ball_id, pos, [0, 0, 0, 1])

# Joint sliders
joint_sliders = {}
joint_limits = {
    0: (-2.897, 2.897),
    1: (-1.762, 1.762),
    2: (-2.897, 2.897),
    3: (-2.017, 2.897),
    4: (-2.897, 2.897),
    5: (-0.017, 2.897),
    6: (-2.897, 2.897),
}
for i in range(len(joint_limits)):
    joint_sliders[i] = p.addUserDebugParameter(f'Joint {i+1}', joint_limits[i][0], joint_limits[i][1], 0)
def update_joints():
    for i in range(len(joint_limits)):
        joint_position = p.readUserDebugParameter(joint_sliders[i])
        p.setJointMotorControl2(franka_id, i, p.POSITION_CONTROL, targetPosition=joint_position)
'''


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

# Detect red ball in image
def detect_red_ball(rgb_img):
    rgb = np.reshape(rgb_img[2], (480, 640, 4))[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    target_pixel = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            target_pixel = (cx, cy)
            cv2.circle(bgr, (cx, cy), 5, (0, 255, 0), -1)

    return target_pixel, bgr, mask

def get_image_jacobian(u, v, z=1.0, fx=600, fy=600):
    """
    Returns the image Jacobian L for a point (u, v)
    z is the estimated depth
    fx, fy are focal lengths (pixels)
    """
    # Here I multiplies the Jacobian by the focal length to keep everything in pixel units, 
    #which is often done in implementation for consistency with measurements in pixels. and
    # to keep everything numerically stabile
    L = np.array([
        [-fx / z, 0, u / z, u * v / fx, -(fx ** 2 + u ** 2) / fx, v],
        [0, -fy / z, v / z, (fy ** 2 + v ** 2) / fy, -u * v / fy, -u]
    ])
    return L

#dof = p.getNumJoints(franka_id) - 1
#joints = range(dof)

# IBVS control
def ibvs_control(target_pixel, image_size=(640, 480)):
    global error_history, v_cam_history

    if target_pixel is None:
        return

    cx, cy = image_size[0] // 2, image_size[1] // 2
    tx, ty = target_pixel
    error = np.array([cx-tx, cy-ty], dtype=np.float32)
    
    # P-controller gain
    Kp =10

    # Full visual velocity (2D error → 6D velocity)
    #z = 0.25  # estimated constant depth to the target
    
    #dynamic depth estimation
    # Get ball position in world (a method that workks only in simulation using some built-in functions in pybullet)
    ball_pos, _ = p.getBasePositionAndOrientation(ball_id)   
    # Get camera pose (EE pose)
    cam_pos, cam_ori = p.getLinkState(franka_id, 11, computeForwardKinematics=True)[:2]
    R_cam = np.array(p.getMatrixFromQuaternion(cam_ori)).reshape(3, 3)
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_cam
    T_wc[:3, 3] = cam_pos  
    # Inverse of camera pose
    T_cw = np.linalg.inv(T_wc)   
    # Homogeneous ball position in world
    ball_h = np.array([*ball_pos, 1.0])
    ball_in_cam = T_cw @ ball_h   
    z = ball_in_cam[2]  # Depth in camera frame

    
    fx, fy = 600, 600
    L = get_image_jacobian(tx - cx, ty - cy, z, fx, fy)
    #L=get_image_jacobian(cx-tx, cy-ty , z, fx, fy)
    try:
        L_inv = np.linalg.pinv(L)  # 6x2 pseudo-inverse
    except np.linalg.LinAlgError:
        return
    
    v_cam_6d = - Kp * L_inv @ error  # camera-frame 6D velocity [vx, vy, vz, wx, wy, wz]
    
    # Store for plotting
    error_history.append(np.linalg.norm(error))
    #error_history.append(error)
    v_cam_history.append(v_cam_6d.tolist())
    
    #print("Visual error:", error)
    #print("v_cam_6d:", v_cam_6d)

    # Get transformation from camera to end-effector (rotation only as the camera is directly attached to the end effector in this simulation) 
    _, cam_quat, _, _, _, _ = p.getLinkState(franka_id, 11, computeForwardKinematics=True)
    R_cam2ee = np.array(p.getMatrixFromQuaternion(cam_quat)).reshape(3, 3)
    R6 = np.block([
        [R_cam2ee, np.zeros((3, 3))],
        [np.zeros((3, 3)), R_cam2ee]
    ])
    v_ee = R6 @ v_cam_6d  # transform to end-effector frame

    # Get the correct 7 joint indices
    controlled_joints = [i for i in range(p.getNumJoints(franka_id)) if p.getJointInfo(franka_id, i)[2] != p.JOINT_FIXED]
    
    # Get joint positions
    joint_states = p.getJointStates(franka_id, controlled_joints)
    joint_positions = [s[0] for s in joint_states]
    zero_vec = [0.0] * len(joint_positions)
    
    # Local position of the point on the end effector (usually [0,0,0] at the link origin)
    local_position = [0, 0, 0]
    
    # Compute Jacobian
    J_lin, J_ang = p.calculateJacobian(franka_id, 11, local_position, joint_positions, zero_vec, zero_vec)
    J = np.vstack((np.array(J_lin), np.array(J_ang)))  # 6x7

    try:
        J_pinv = np.linalg.pinv(J)  # 7x6
        q_dot = J_pinv @ v_ee       # joint velocities
    except np.linalg.LinAlgError:
        return
    
    
    #print("Jacobian J (6x7):\n", J)
    #print("Pseudo-inverse of Jacobian (J_pinv):\n", J_pinv)
    #print("L_inv (image Jacobian inverse):\n", L_inv)
    #print("q_dot:", q_dot)
    
    # Send joint velocity commands
    controlled_joints = [i for i in range(p.getNumJoints(franka_id)) if p.getJointInfo(franka_id, i)[2] != p.JOINT_FIXED]
    p.setJointMotorControlArray(
    bodyUniqueId=franka_id,
    jointIndices=controlled_joints,
    controlMode=p.VELOCITY_CONTROL,
    targetVelocities=q_dot.tolist()
)






home_position = [0, -0.4, 0, -1.8, 0, 1.4, 0] # Set to a neutral/home configuration


# Move the robot to the home position
for i in range(len(home_position)):
    p.setJointMotorControl2(franka_id, i, p.POSITION_CONTROL, targetPosition=home_position[i])

# Simulate for a few steps to let the robot settle into the home position before running the main simulation loop
for _ in range(100):
    p.stepSimulation()
    time.sleep(0.01)

# Random target position
target_pos_ball = [random.uniform(0.3, 0.7), random.uniform(-0.4, 0.4), 0.02]
p.resetBasePositionAndOrientation(ball_id, target_pos_ball, [0, 0, 0, 1])
'''
#randomizing the objects
target_pos_box1 = [random.uniform(0.3, 1), random.uniform(-0.5, 0.5), 0]
p.resetBasePositionAndOrientation(box_id1, target_pos_box1, [0, 0, 0, 1])

target_pos_box2 = [random.uniform(0.3, 1), random.uniform(-0.5, 0.5), 0]
p.resetBasePositionAndOrientation(box_id2, target_pos_box2, [0, 0, 0, 1])
'''

# For diagnostics
error_history = []
v_cam_history = []
# Initialize the file to save data
data_file = "simulation_data.csv"
with open(data_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Time Step', 'Error', 'v_x', 'v_y', 'v_z', 'w_x', 'w_y', 'w_z'])  # Header

'''#initiating the diffusion model....
img = panda_camera() #current frame
#Convert image to PIL format and resize it to 256x256 before feeding into the diffusion model
rgb = np.reshape(img[2], (480, 640, 4))[:, :, :3]
pil_img = Image.fromarray(rgb).resize((256, 256))
# Predict the next frame using InstructPix2Pix
prompt = "reach the red ball"
num_inference_steps = 100
image_guidance_scale = 1.5
guidance_scale = 7.5
predicted_frame = pipe(
    prompt,
    image=pil_img,
    num_inference_steps=num_inference_steps,
    image_guidance_scale=image_guidance_scale,
    guidance_scale=guidance_scale,
    generator=generator
).images[0]'''
data_dir = "realtime-test"
os.makedirs(data_dir, exist_ok=True)
# Main loop
i = 0
while True:
#while i<5:
    #p.stepSimulation()
    #for _ in range(10):
    p.stepSimulation()
    #update_joints()
    #update_ball_position() #BALL SLIDERS
    img = panda_camera() #current frame
    #Convert image to PIL format and resize it to 256x256 before feeding into the diffusion model
    rgb = np.reshape(img[2], (480, 640, 4))[:, :, :3]
    pil_img = Image.fromarray(rgb).resize((256, 256))
    pil_img.save(os.path.join(data_dir, f"frame_{i:02d}.png"))
    image= Image.open(os.path.join(data_dir, f"frame_{i:02d}.png")).convert("RGB")
    # Predict the next frame using InstructPix2Pix
    prompt = "reach the red ball"
    num_inference_steps = 100
    image_guidance_scale = 1.5
    guidance_scale = 7.5
    predicted_frame = pipe(
        prompt,
        image=image,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    # Convert predicted image back to OpenCV (BGR) for display
    edited_cv_img = cv2.cvtColor(np.array(predicted_frame), cv2.COLOR_RGB2BGR)
    current_cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Show current and predicted frames
    cv2.imshow("Current Frame", current_cv_img)
    cv2.imshow("Predicted Goal Frame", edited_cv_img)

    #ball_pixel, bgr_img, mask = detect_red_ball(img) -> REPLACED BY THE DIFFUSION MODEL To find the next frame predicted view
    #print(ball_pixel)
    #ibvs_control(ball_pixel) -> REPLACED BY THE RTVS
    
    
    #RTVS start
    img_src = rgb
    #img_src =np.array(Image.fromarray(rgb).resize((256, 256)))
    pre_img_src = img_src.copy()
    predicted_frame_resized = predicted_frame.resize((640, 480))
    img_goal = np.array(predicted_frame_resized)
    #img_goal=np.array(predicted_frame)
    photo_error_val = mse_(img_src, img_goal)
    perrors = [photo_error_val]
    print("Initial Photometric Error: ")
    print(mse_(img_src, img_goal))
    
    start_time = time.time()
    step = 1
    rtvs = Rtvs(img_goal)
    #while photo_error_val > 200 and step < 30:
    while photo_error_val > 400 and step < 60:
        stime = time.time()
        #vel = call_rtvs(img_goal, img_src, pre_img_src)
        vel = rtvs.get_vel(img_src, pre_img_src)
        vel_adj = np.array([
                    vel[0],        # move_right:  +V[0]
                    -vel[1],       # move_up:     -V[1]
                    -vel[2],       # move_backward: -V[2]
                    vel[3],        # look_up:     +V[3]
                    -vel[4],       # look_left:   -V[4]
                    vel[5],        # look_anti:   +V[5]
                ])
                        #vel=[0,0,0,0,0,0]
        algo_time = time.time() - stime
        photo_error_val = mse_(img_src, img_goal)
        perrors.append(photo_error_val)

        #start of pybullet franka arm motion integration 
        v_cam_6d = vel_adj  # camera-frame 6D velocity [vx, vy, vz, wx, wy, wz]
        # Get transformation from camera to end-effector (rotation only as the camera is directly attached to the end effector in this simulation) 
        _, cam_quat, _, _, _, _ = p.getLinkState(franka_id, 11, computeForwardKinematics=True)
        R_cam2ee = np.array(p.getMatrixFromQuaternion(cam_quat)).reshape(3, 3)
        R6 = np.block([
            [R_cam2ee, np.zeros((3, 3))],
            [np.zeros((3, 3)), R_cam2ee]
        ])
        #v_ee = R6 @ v_cam_6d  # transform to end-effector frame
        scale = 1
        v_ee = scale * (R6 @ v_cam_6d)
        
        # Get the correct 7 joint indices
        controlled_joints = [i for i in range(p.getNumJoints(franka_id)) if p.getJointInfo(franka_id, i)[2] != p.JOINT_FIXED]       
        # Get joint positions
        joint_states = p.getJointStates(franka_id, controlled_joints)
        joint_positions = [s[0] for s in joint_states]
        zero_vec = [0.0] * len(joint_positions)      
        # Local position of the point on the end effector (usually [0,0,0] at the link origin)
        local_position = [0, 0, 0]       
        # Compute Jacobian
        J_lin, J_ang = p.calculateJacobian(franka_id, 11, local_position, joint_positions, zero_vec, zero_vec)
        J = np.vstack((np.array(J_lin), np.array(J_ang)))  # 6x7
        try:
            J_pinv = np.linalg.pinv(J)  # 7x6
            q_dot = J_pinv @ v_ee       # joint velocities
        except np.linalg.LinAlgError:
            continue
        
        # Send joint velocity commands
        p.setJointMotorControlArray(
        bodyUniqueId=franka_id,
        jointIndices=controlled_joints,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocities=q_dot.tolist())
        
        #for _ in range(2):
        p.stepSimulation()

        #end of pybullet franka arm motion integration 
        

        print("Step Number: ", step)
        print("Velocity : ", vel.round(8))
        print("Photometric Error : ", photo_error_val)
        #print("algo time: ", algo_time)

        pre_img_src = img_src
        img = panda_camera() #current frame
        rgb = np.reshape(img[2], (480, 640, 4))[:, :, :3]
        img_src = rgb
        
        #pil_img = Image.fromarray(rgb).resize((256, 256))
        #current_cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        #cv2.imshow("Live Frame", current_cv_img)

        step = step + 1  
    #RTVS end
   
    
   
    
   
   # Display
    #cv2.imshow("Camera View", bgr_img)
    #cv2.imshow("Red Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if len(error_history) > 0 and len(v_cam_history) > 0:
        with open(data_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i, error_history[-1]] + list(v_cam_history[-1]))


    
    i += 1
    time.sleep(0.01)

cv2.destroyAllWindows()
p.disconnect()