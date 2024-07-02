# Intelligent Robotic Navigation

## Authors
- Sai Jagadeesh Muralikrishnan
- Varun Lakshmanan

## Introduction
This project aims to enhance robotic navigation in dynamic environments using advanced deep reinforcement learning techniques. Specifically, we implemented a Double DQN with a Dueling architecture to improve the performance of a TurtleBot Burger in the Gazebo simulation environment. This approach addresses overestimation bias and stability issues inherent in the original DQN algorithm.

## Abstract
The goal of this project is to improve robotic navigation with advanced deep reinforcement learning techniques in environments with dynamic obstacles. By implementing a Double DQN with a Dueling architecture, we enhanced the performance of a TurtleBot Burger in the Gazebo simulation environment. This method improves navigation efficiency and reduces the number of training episodes needed.

## Index Terms
- DQN
- Double DQN
- Dueling DQN
- ROS2
- TurtleBot3

## Literature Review
Our project builds on the following key references:
1. Tomas Vrana's GitHub repository "A ROS2 framework for DRL autonomous navigation on mobile robots with LiDAR".
2. Mohit Sewak's book "Deep Q Network (DQN), Double DQN, and Dueling DQN: A Step Towards General Artificial Intelligence" (2019).
3. Ziyu Wang et al.'s paper "Dueling network architectures for deep reinforcement learning" (ICML 2016).

## Methodology
### Design and Approach
We used the Gazebo simulation environment and ROS2 for communication and control of the TurtleBot Burger robot. The methodology includes:
- Vanilla DQN Algorithm
- Dueling Double DQN Algorithm

### Tools and Procedures
- **Framework**: A ROS2 Framework for DRL Autonomous Navigation
- **Libraries Used**: Numpy, PyTorch

### Setup of the Simulation Environment
1. ROS2 Foxy and Gazebo on Ubuntu.
2. Configured TurtleBot Burger model with LiDAR and odometry sensors.
3. Deployed a custom world with dynamic obstacles in Gazebo.

### Implementation and Training of the Algorithms
1. Cloned the GitHub repository for initializing the Gazebo world.
2. Implemented Double DQN and Dueling DQN algorithms.
3. Trained models using data gathered from the simulation environment.

## Evaluation of the Models
We evaluated the performance of the models through rigorous testing and analysis, comparing the vanilla DQN with the proposed Dueling Double DQN.

## Contributions
### Problems with Vanilla DQN
- Overestimation Bias
- Training Instability

### Improvements with Dueling Double DQN
- Enhanced Stability
- Improved Efficiency
- Better Policy Development

## Experimental Results
### Training Phase - 1
During the first phase of training, we aimed to train the models from 0 to 1700 episodes to assess their performance.

#### Vanilla DQN Performance
The performance of the Vanilla DQN model showed moderate results through the training at 1700 episodes. The model was less efficient compared to the Dueling method, with the robot’s success rate in reaching the goal only gradually increasing. The robot encountered frequent collisions with walls and dynamic obstacles. Moreover, the average rewards over every 10 episodes remained below zero, indicating a negative performance trend.

#### Dueling Double DQN Performance
In contrast, the Dueling Double DQN model demonstrated a steep improvement in performance. The success rate of the robot in reaching the goal steadily increased, and the number of collisions significantly decreased. By 1700 episodes, the rewards had already progressed to positive values. These observations underscore the enhanced efficiency and capability of the Dueling Double DQN model in navigating complex environments with higher precision and fewer errors.

### Training Phase - 2
Both models were reset and started from episode 0 to verify the performance observed in Phase 1 and to acquire fresh results by the end of Phase 2. The training time for Vanilla DQN from the 0th episode to the 3000th episode took approximately 28 hours, while Dueling Double DQN completed the training in about 26 hours. This phase was designed to test the model's performance after 3000 episodes of training.

#### Comparison After 3000 Episodes
- **Vanilla DQN**: The performance of the Vanilla DQN showed that the success rate started increasing gradually above the collision rate, but the increase was not significant. After 1700 episodes, there was a dramatic increase in success, but the collision rate with the robot also increased. The average rewards after 3000 episodes for Vanilla DQN remained below zero, in the negative side, still not matching the performance of the Phase 1 Dueling Double DQN model.
- **Dueling Double DQN**: This model performed beyond expectations where the collision rate drastically decreased while the success rate significantly increased. Additionally, the rewards were maintained above zero with positive values.

### Extended Training
An additional 2000 episodes were trained for the Dueling Double DQN, showing a 75% success rate in complex environments. This extensive testing phase produced positive results. The effectiveness of the Dueling Double DQN model to reduce collisions and increase successful navigation to the objective was noticeably improved.

### Testing After 3000 Episodes
Both models were tested over the next 20 episodes in a more complex environment than what they were trained on previously. This new environment featured 6 moving obstacles instead of 4, with increased space inside the map to accommodate the additional dynamic obstacles and the burger bot.
- **Vanilla DQN**: The burger bot reached the goal points but not consistently, reflecting a lower success rate as it collided with walls and dynamically moving obstacles most of the time, leading to an increased rate of timeouts.
- **Dueling Double DQN**: The robot effectively navigated to randomly spawning goal points in every episode, occasionally hitting dynamic obstacles but rarely colliding with wall obstacles and experiencing very few timeouts.

## Simulation Videos and Images
### Training Phase - 1 Results
<!-- Add simulation videos, GIFs, and images here -->
[Link to Training Phase - 1 Video](https://drive.google.com/file/d/1qH-McJhEq93v7Rdqwxw6BZ8j4bVdsPM-/view)
![image](https://github.com/cravotics/Intelligent-Robotic-Navigation/assets/90138418/46d81f6f-bad9-46a8-9e62-4ccbdbec4ce5)
![image](https://github.com/cravotics/Intelligent-Robotic-Navigation/assets/90138418/55be9366-41c3-44d6-b7ed-329fad1ab66f)


### Training Phase - 2 Results
<!-- Add simulation videos, GIFs, and images here -->
[Link to Training Phase - 2 Video](https://drive.google.com/file/d/1tsFBgujWl74BMfZTGFvLvy3U1bwCoKbR/view)

![image](https://github.com/cravotics/Intelligent-Robotic-Navigation/assets/90138418/a50eb866-e3a8-446c-8cf3-cfcfb9735ddd)
![image](https://github.com/cravotics/Intelligent-Robotic-Navigation/assets/90138418/ad781ed1-0718-4566-a398-5fe3dbf6ca7c)

### Map Used 
![image](https://github.com/cravotics/Intelligent-Robotic-Navigation/assets/90138418/d53c7b5a-0720-4ba9-a082-cec73d425807)

### Extended Training Results
<!-- Add simulation videos, GIFs, and images here -->
[Link to Extended Training Video](https://drive.google.com/file/d/1NFG_7DQoTRUQjtM-QYpQCgTMvu_C78V3/view)

### Testing Results
<!-- Add simulation videos, GIFs, and images here -->
[Link to Testing Video](https://drive.google.com/file/d/12PErRUwAVW8FRP7aUm8Ux1sWXIeSSM-0/view)
![download](https://github.com/cravotics/Intelligent-Robotic-Navigation/assets/90138418/475d092c-1f9d-4b39-a210-6f1543f132a6)

## Problems Faced
1. **GPU Training Interruptions**: Our initial attempts to train the model using the GPU were met with unexpected system issues, causing the training to stop before completion. We decided to proceed without the GPU, which increased the training duration but enabled us to achieve the planned outcomes.
2. **Debugging the Dueling Double DQN**: Debugging the modified algorithm was time-consuming and required closely examining the algorithm’s performance and its interactions within the environment. These efforts were crucial in refining the model and ensuring its robustness.

## Conclusion
The Dueling Double DQN algorithm significantly outperforms the Vanilla DQN in dynamic environments, providing a robust foundation for future advancements in robotic navigation. The modified algorithm demonstrated improved stability, efficiency, and better policy development, resulting in higher success rates and reduced collision incidents.

## References
1. Tomas Vrana, “A ROS2 Framework for DRL Autonomous Navigation on Mobile Robots with LiDAR,” GitHub repository.
2. Mohit Sewak, “Deep Q Network (DQN), Double DQN, and Dueling DQN: A Step Towards General Artificial Intelligence,” 2019.
3. Ziyu Wang et al., “Dueling network architectures for deep reinforcement learning,” ICML 2016.
4. H. van Hasselt et al., “Deep reinforcement learning with double Q-Learning,” AAAI 2016.

## Contact
For any queries, please contact us at:
- Sai Jagadeesh Muralikrishnan: [jagkrish@umd.edu](mailto:jagkrish@umd.edu)
- Varun Lakshmanan: [varunl11@umd.edu](mailto:varunl11@umd.edu)
