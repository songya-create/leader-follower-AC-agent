from tqdm import tqdm
from agent import Agent
from common.buffer import ReplayBuffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = Agent(self.args)

        self.buffer = ReplayBuffer(self.args)
        if self.args.evaluate==False:
         self.writer_reward = SummaryWriter(self.args.algorithm + "logs/add_scalar3")

    def m_matrix(self):
        matrix = np.eye(self.args.n_agents)
        #设置相邻值为一代表这相邻编号的智能体2相连
        for j in range(self.args.edge_n+1):
            for i in range(self.args.n_agents):
              if i + j < self.args.n_agents:
                matrix[i][i + j] = 1
                matrix[i + j][i] = 1
                matrix[i][(self.args.n_agents+i- j)%self.args.n_agents] = 1
                matrix[(self.args.n_agents + i - j) % self.args.n_agents][i] = 1

        if self.args.algorithm == "GRUActor":
            matrix=np.ones((self.args.n_agents,self.args.n_agents));
        return matrix
    def run(self):

        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                self.args.shape = np.random.randint(0, 2)
                s ,_= self.env.reset()###GAT每个智能体当前观测值只有自身的状态，获取其他智能体的状态是通用图注意力完成的
            self.env.render()
            matrix = self.m_matrix()
            with torch.no_grad():
                actions= self.agents.select_action(s,matrix,self.epsilon)
            u=actions
            s_next, r, done, info,collis_n,arrive,abs_dis,_ = self.env.step(actions,self.args.shape)  #actions:[上下移动量,左右移动量,剩下3个为通信数据这里用不到]

            next_matrix=self.m_matrix()
            self.buffer.add(s, u, r, s_next,matrix,next_matrix,done)
            s = s_next
            if self.buffer.current_size >=self.args.batch_size  and time_step % 10 == 0:
               batch_buffer=self.buffer.getBatch(self.args.batch_size)
               self.agents.learn(batch_buffer)
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                self.evaluate()
            if time_step > 0 and time_step % 2000 == 0 and self.buffer.pointer>400:
                printr=sum(sum(self.buffer.rewards[self.buffer.pointer-400:self.buffer.pointer,:]))/self.buffer.rewards[0:400,:].size
                self.writer_reward.add_scalar('reward',printr, int(time_step / 2000))
            self.noise = max(0.001, self.noise - 0.0000001)
            self.epsilon = max(0.001, self.epsilon - 0.0000001)

           # np.save('my_buffer.npy', self.buffer.buffer)
           # load_dict = np.load('my_buffer.npy', allow_pickle=True).item()


    def evaluate(self):

        returns = []
        sum_collis_count = []
        sum_dis_D=[]
        succed_count=0
        terminal_t=0
        formation_dis=0
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s ,_= self.env.reset()
            rewards = 0
            arrive_flag = 0
            collis_count=0
            dis_D=0
            arrive_count=0
            matrix = self.m_matrix()
            for time_step in range(self.args.evaluate_episode_len):
                    self.env.render()

                    with torch.no_grad():
                       actions = self.agents.select_action(s, matrix, self.epsilon)
                    s_next, r, done, info, collis_n, arrive, abs_dis,_= self.env.step(actions,self.args.shape)  # actions:[上下移动量,左右移动量,剩下3个为通信数据这里用不到]
                    s = s_next
                    if  time_step >350:#在最后50步时测评距离差
                        dis_D+=sum(abs_dis)/self.args.agents_num
                    formation_dis +=(sum(abs_dis)-abs_dis[0])/(self.args.agents_num-1)
                    rewards += sum(r)/self.args.agents_num
                    collis_count +=collis_n
                    arrive_count += arrive
                    if arrive_count > 4 and arrive_flag==0:
                         terminal_t += time_step
                         arrive_flag =1
            if arrive_flag ==0:
                terminal_t += time_step
            print(collis_count)
            sum_collis_count.append(collis_count)
            sum_dis_D.append(dis_D/50)
            if arrive_count>10 and collis_count<1:
                succed_count +=1
            returns.append(rewards/ self.args.evaluate_episode_len)
            print("奖励",rewards/ self.args.evaluate_episode_len)
        self.env.close()

        print('测试次数',self.args.evaluate_episodes)
        print('每个episode平均碰撞次数', sum(sum_collis_count) / self.args.evaluate_episodes)
        print('智能体最后50步与目标距离差的均值', sum(sum_dis_D) / self.args.evaluate_episodes)
        print('成功率', succed_count/self.args.evaluate_episodes)
        print('到达终点花费时间',terminal_t/self.args.evaluate_episodes)
        print('编队完成能力formation_dis', formation_dis / (self.args.evaluate_episodes*400))


        return sum(returns) / self.args.evaluate_episodes, sum(sum_collis_count) / self.args.evaluate_episodes,  succed_count/ self.args.evaluate_episodes




