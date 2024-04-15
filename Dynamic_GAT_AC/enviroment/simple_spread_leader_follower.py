import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import math
import cv2
import numpy as np
import  random

class KalmanFilter:
    kf = cv2.KalmanFilter(4, 4)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def predict(self, coordX, coordY,vx,vy):
        '''This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)],[np.float32(vx)],[np.float32(vy)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        #x, y = predicted[0], predicted[1]
        return predicted[0], predicted[1],predicted[2], predicted[3]

kf0 = KalmanFilter()
kf1 = KalmanFilter()
kf2 = KalmanFilter()
img = cv2.imread("E:/code_project/code_py_work/MADDPG_mpe_mutigoal/MADDPG-master/predicte/img_1.png")
img = cv2.resize(img, (1200, 1200))
class Scenario(BaseScenario):
    def make_world(self,num_agents):
        world = World()
        world.visionable_bound=0.25
        # set any world properties first
        world.dim_c = 2

        num_landmarks=num_agents
        world.num_agents =num_agents
        world.num_landmarks =  num_landmarks #目标是几变形就设置几个，目的是覆盖目标并形成编队形状

        # 修改二增加障碍物
        num_obstruction = 0 #障碍物设置，只编队控制不增加障碍物，如果后期要增加用来训练智能体适应周围有障碍物的状态
        world.collaborative = True
        world.myformation=[[0,0] for i in range(num_agents)]
        world.shape=0



        # add agents
        world.agents = [Agent() for i in range(num_agents)] #创建智能体
        for i, agent in enumerate(world.agents):
            agent.u_noise = 0
            agent.max_speed=0.1
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.01
            agent.num=i

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = True
            landmark.size = 0.005
            landmark.num = i
        world.obstruction = [Landmark() for i in range(num_obstruction)]
        for i, obstruction in enumerate(world.obstruction):
            obstruction.name = 'obstruction %d' % i
            obstruction.collide = False
            obstruction.movable = True
            obstruction.size  = 0.04 #静态0.2  #动态 obstruction.size = 0.04



        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])

            agent.posx = np.zeros(400)
            agent.posy =  np.zeros(400)
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.8, 0.25, 0.25])
        for i, obstruction in enumerate(world.obstruction):
            obstruction.color = np.array([0.1,0.1,0.1])
        # set random initial states
        for i, agent in enumerate(world.agents):
            #修改一固定初始位置
            #agent.state.p_pos=np.array([0.9-0.3*i,-0.88])
            agent.state.p_pos = np.random.uniform(-0.9, -0.1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        centerpos=np.random.uniform(-0.5, 0.5, 2)
        centerpos=(0,0)

        '''
        landmark_state = self.calculate_regular_polygon_vertices(centerpos[0],centerpos[1], 0.2+ world.num_landmarks*0.01, world.num_landmarks)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.array(landmark_state[i])
            # agent.state.p_pos = np.random.uniform(-0.9, -0.1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            '''
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos =  centerpos
            # agent.state.p_pos = np.random.uniform(-0.9, -0.1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        self.time_t = 0

        for i, agent in enumerate(world.agents):

            agent.E_rew = [0, 0]
            dists = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[agent.num].state.p_pos)))
            agent.E_rew[0]=-dists



    def calculate_circle(self,center_x, center_y, radius, num_sides):
        vertices = []
        for i in range(num_sides):
            angle = (2 * math.pi * i) / num_sides
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            vertices.append((x, y))
        return vertices
    def calculate_line(self,center_x, center_y, kong, num_sides):
        vertices = []
        for i in range(num_sides):
            x = center_x + kong*int((i+1)/2)*((-1)**i)
            y = center_y
            vertices.append((x, y))
        return vertices

    def formation_conntrol(self,leader,shape,number):
        if shape==0:#shape=="circle":
            return   self.calculate_circle(leader[0]- (0.2+number*0.01),leader[1],  0.2+ number*0.01, number)
        else:#shape=="line":
            return self.calculate_line(leader[0],leader[1],(0.1+number*0.01), number)



    # 计算reward以及相撞
    def benchmark_data(self, agent, world,shape):
        rew= 0
        collisions = 0
        collisions_obs=0
        occupied_landmarks = 0
        min_dists = 0
        done=False
        world.shape=shape
        if agent.num==0:
           agent.state.p_pos = world.landmarks[0].state.p_pos####formation test
           world.myformation=self.formation_conntrol(agent.state.p_pos,shape,world.num_agents)
           dists = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos)))##agent0为leader，追寻目标
           rew-=dists
        else:
           #与目标点距离的奖励，惩罚和占领目标点会获得奖励
           dists = np.sqrt(np.sum(np.square(agent.state.p_pos - world.myformation[agent.num])))##其他跟踪leader

        #agent.E_rew[1] = np.exp(-(abs_dis)) - (abs_dis)  # 构建势能场
        agent.E_rew[1]=dists

        #rew +=(agent.E_rew[0]-agent.E_rew[1])
        rew += 2*np.exp(-(4 * dists))
        agent.E_rew[0] = agent.E_rew[1]
        if dists == 0:
            rew+=1
        if dists < agent.size:
           occupied_landmarks += 1
        # rew= max(1.4-dists,0)
        #不能撞墙但不记录撞击次数
        #if self.is_coll_wall(agent)[0]:
            #collisions += 1
        rew -=np.exp(6*(self.is_coll_wall(agent)[1]-2))
        #智能体之间不能互相撞击
        if agent.collide:
            for i,a in  enumerate(world.agents):
                if not(a.num ==agent.num):
                    rew -= self.is_collision(a, agent)[1]
                    if self.is_collision(a, agent)[0]:
                        collisions += 1
            """
            for i, obstruction in enumerate(world.obstruction):
                coll_obs= self.is_collision(obstruction, agent)
                rew -=coll_obs[1]
                if coll_obs[0]:
                    collisions_obs += 1
                    #rew -= 1
            """
        agent.v_last=agent.state.p_vel
        #rew-=np.sqrt(np.sum((np.square(agent.state.p_vel-world.agents[0].state.p_vel))))#速度一致性
        return (rew, collisions,occupied_landmarks,dists)#, min_dists, occupied_landmarks)
    
    
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        if dist < dist_min:
            return [True,1]
        if  dist_min<=dist and dist < 5*dist_min:###动态这里是dist <  3*dist_min
            return [False,np.exp(-10*dist)]
        else:
            return  [False,0]
    #撞击墙壁惩罚
    def is_coll_wall(self,a):

        if  all(abs(i) < 1 for i in a.state.p_pos):
            return [False,sum(abs(i)for i in a.state.p_pos)]##第二个参数可以代表里边缘的间距
        else:
            return [True,1]





    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew


    def updat_obstruction(self,world):
        self.time_t+=1
        pre0=[]
        pre1=[]
        pre2=[]
        #无障碍物的时候，这里不执行
        for i, obstruction in enumerate(world.obstruction):
            noise = 0.5*np.random.randn(2)  # gaussian noise
            obstruction.state.p_pos+=world.dt*obstruction.state.p_vel##np.random.uniform(-0.8,+0.8, world.dim_p)
            Vx=(obstruction.vbase)*(1)*((-1)**i)*((-1)**(int(self.time_t/150)))
            #Vy=0.02*math.cos(0.03*self.time_t)
            Vy=0.003*(1+noise[1])
            #Vy=0.02*((-1)**i)*((-1)**(int(self.time_t/200)))
            obstruction.state.p_vel= np.array([Vx,Vy])
        #目标位置不发生移动
        world.landmarks[0].state.p_pos += world.dt*world.landmarks[0].state.p_vel
        noise =0.5*np.random.randn(2)
        Vx = -0.015*(1) * ((-1) ** i) * ((-1) ** (int(self.time_t / 1000)))
        # Vy=0.02*math.cos(0.03*self.time_t)
        #Vy = 0.005*(1+noise[1])*math.cos(0.01*self.time_t)
        #Vy = 0.015 * (noise[1])
        world.landmarks[0].state.p_vel = np.array([Vx,Vy])
           
    def observation(self, agent, world):
       # self.get_path(agent,times = self.time_t)
        #if self.time_t==399:
        #    print("----this is agent",agent.name)
         #   print(agent.posx)
          #  print(agent.posy)


       ##如果要更新障碍物体状态就在这里实现
        #if agent.name=='agent 1':
        #self.updat_obstruction(world)
        # get positions of all entities in this agent's reference frame

         ##智能体的状态信息
         #[当前智能体节点的坐标，目标节点的坐标，观测到的局部观测数据]
        if agent.num==0:
          target_pos = world.landmarks[agent.num].state.p_pos
        else:
           target_pos = world.myformation[agent.num]
        #这里有个隐含的操作就是输入的所见局部范围内的障碍物不能超过·3个
        """
        max_suround_obs=3
        obs_pos = np.zeros((max_suround_obs, 2))
        i=0
        for entity in world.obstruction :  # world.entities:
            if self.visionable_area(world,entity):
                obs_pos[i]=entity.state.p_pos
                i = i + 1 if i < max_suround_obs-1 else 0 ##如果超出了定义环绕个数的限制，就去替换第一个观测到的数据
        """
        # 这里有个隐含的操作就是输入的所见局部范围内的其他智能体不能超过·6个/9个(不考虑障碍物)
        max_suround_agent=9
        other_pos = np.zeros((max_suround_agent, 2))
        i=0
        for entity in world.agents:
            if not (entity .num == agent.num):
                if self.visionable_area(world, entity,agent):
                    other_pos[i] = entity.state.p_pos
                    i = i + 1 if i<max_suround_agent-1 else 0

        target_speed=world.agents[0].state.p_vel ##速度一致性一共考虑速度一致性和位置一致性
        #多个数组沿一个指定的轴方向进行拼接
        return np.concatenate([agent.state.p_pos] +[agent.state.p_vel] +[target_pos]+ [other_pos.reshape(-1)] )#+[obs_pos.reshape(-1)])


    def visionable_area(self,world, a_vis,agent):
                if np.sqrt(np.sum(np.square(agent.state.p_pos-a_vis.state.p_pos))) <world.visionable_bound:
                  return True
                else:
                  return False

    def get_path(self, agent,times):
        agent.posx[times] = agent.state.p_pos[0]
        agent.posy[times] = agent.state.p_pos[1]
        return agent.posx,agent.posy