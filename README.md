# Intelligent Visual Servoing with Diffusion Models

## Overview

It was known for a long time in the robotics field that autonomous robotic systems follows a three-steps architecture that separate the perception, planning, and control during execution. Such traditional architectures usually require a predefined goal location or image and depend on a considerable number of sensors for feedback making them limited in adaptability and scalability. For instance, in a lot of real-world scenarios, robots need to perform tasks in environments where the final goal state is unknown at deployment, or where the robot must interpret high-level instructions rather than simply matching a reference location or image. These constraints hinder the deployment of robots in unseen and dynamic environments.

With the rise of the new deep learning models recently, a new robotic architecture called action models was introduced to overcome the mentioned limitations by leveraging recent advances in generative models, specifically the diffusion-based image editing models. The new aarchitecture merges both perception and planning into a single step to enable robots to perform long-range navigation and manipulation tasks starting from arbitrary initial states without the need for predefined goal locations or images and with a minimal sensor suite, such as a single camera mounted on the robot base or end-effector. This is done by synthesizing intermediate goal images from high-level language prompts and current sensory observations. Overcoming such limitations was the motivation behind choosing that 2-step framework. The specific pipeline choice is [Imagine2Servo](https://arxiv.org/abs/2410.12432) which focuses on performing Visual Servoing intelligently with diffusion-driven goal generation for robotic tasks.

<div align="center">
<img src = "media/2_steps.png" width="50%">
<p>The Robotic system 2-step architecture
</p>
</div>

## Imagine2Servo Architecture

Imagine2Servo paper deals with the task of generating the next subgoal for the servoing controller as editing the pixels of the current input image. Given the current image and task description P, authors aim to generate the subgoal image using the foresight diffusion model. They employ RTVS (Real-Time Visual Servoing) as the Image-Based Visual Servoing (IBVS) controller to reach the subgoals predicted by the foresight module. They use the RTVS servoing algorithm without any fine-tuning to the newer environments.

<div align="center">
<img src = "media/imagine2servo.png" width="100%">
<p>Imagine2Servo architecture
</p>
</div>

The Foresight Model serves as the generative part of the framework, producing sub-goal images that guide the robot to complete high-level tasks specified by language prompts. The process begins with a raw text prompt, such as "reach the red ball" which is transformed into a dense semantic embedding by a text encoder. This embedding, combined with a noisy latent vector and conditional image latents representing the robot’s current observation, forms the input to a denoising diffusion process implemented via the U-Net architecture. The U-Net iteratively refines the noisy latent to generate a sub-goal image. Finally, an image decoder translates the refined latent into an actual image, representing the next intermediate state the robot should achieve to complete its the task. The authors used Instructpix2pix image editing framework in this step.

The Visual servoing loop converts the generated sub-goal image into actionable control commands. This loop receives the sub-goal (goal image), the robot’s current camera observation, and the previous image as inputs. Using FlowNet2.0, it estimates the optical flow between the goal and current images, determining the pixel-wise motion required to move toward the sub-goal. 

## Implementation & Key Contributions

In the implementation of the Imagine2Servo framework, we began by designing the simulation setup using PyBullet smulator, where a Franka Emika Panda 7-DOF manipulator equipped with a single eye-in-hand camera was used to perform a range of reaching tasks. To support robust training and evaluation, we generated a comprehensive dataset consisting of variations of reaching tasks such as “reach the red ball” and “reach the blue box” with each variation containing a number of demonstrations and five camera frames recorded per demonstration as ground truth sub goals with randomized initial frames to ensure diversity. A key contribution in our pipeline was the replacement of the traditional FlowNet2 optical flow network with NeuFlow2, a more performant model on edge devices, to improve the accuracy and stability of visual servoing in the RTVS loop. Implementation of the diffusion-driven goal generation pipeline, as described in the original paper, was done. This pipeline removes the dependency on predefined goal images and allows for more adaptive and language-driven robotic behavior. We fine-tuned the InstructPix2Pix diffusion model on our custom dataset, focusing on single-view camera input to mimic realistic robotic perception scenarios. The entire system is fully integrated, including simulation, diffusion model, and a closed-loop visual servoing loop, resulting in a reproducible and extensible architecture.

<div align="center">
<img src = "media/modified_architecture.png" width="100%">
<p>Imagine2Servo architecture
</p>
</div>

## Results

<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/93aa0815-77b7-4e13-9c53-4111206d6497" width="250px"/><br/>
        <sub>Reaching red ball task - Test 1</sub>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/a11d914e-a454-476b-91d6-f3bf2a632ffa" width="250px"/><br/>
        <sub>Reaching red ball task - Test 2</sub>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/e30a1b33-4a4f-4a6e-9588-ca68e379da69" width="250px"/><br/>
        <sub>Reaching red ball task - Test 3</sub>
      </td>
    </tr>
  </table>
</p>

## How to Use


