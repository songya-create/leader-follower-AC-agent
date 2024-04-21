# The leader-follower-framework based formation control algorithm, combined with a multi-agent reinforcement learning algorithm

## task description

This is a scalable leader-follower formation control framework.

Scalable Learning: Our framework demonstrates scalability, allowing the knowledge gained from training with 6 AGVs to effortlessly extend to formations involving varying numbers of AGVs, ranging from 3 to 12.

Versatile Formation Shapes: We showcase the adaptability of our approach by successfully addressing formation tasks with not only basic shapes like lines and circles but also more intricate shapes such as S and V shapes. This versatility enhances the applicability of our framework to a wide range of real-world scenarios.


![trac_c](https://github.com/songya-create/leader-follower-AC-agent/assets/63812791/4a75e9fd-5736-46ec-982f-71af105ffecf)
![trac_l](https://github.com/songya-create/leader-follower-AC-agent/assets/63812791/59722b00-1cea-40e3-a63f-1decfc1df96d)
![trac_12c](https://github.com/songya-create/leader-follower-AC-agent/assets/63812791/795d141e-a11a-4414-9354-1a58e78c5936)
![trac_12l](https://github.com/songya-create/leader-follower-AC-agent/assets/63812791/d69b377b-b953-461a-8c74-1870932af284)

## algorithm description

Firstly, we establish a formation shape generator by integrating the leader-follower algorithm. This enables us to ensure coordinated movement of AGVs within the formation.

Secondly, we design an action output controller by incorporating a multi-agent reinforcement learning algorithm. This controller empowers AGVs to make dynamic decisions based on environmental cues and desired formation shapes.

Thirdly, to enhance the adaptability of the actor-critic network, we employ a graph attention mechanism. This mechanism allows AGVs to focus on relevant environmental information crucial for decision-making.

## Requirements

- python=3.7
- [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs)
- torch=1.12.0(cud11

## Quick Start

Directly run the main.py

## Note
