import torch
import os
from D_GAT_AC.actor_critic import Actor, Critic,CommCritic,leaderActor,CommActor,BaseActor,BaseCritic
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class D_GAT_AC:
    def __init__(self, args):
        self.args = args
        if self.args.evaluate == False:
            self.writer1 = SummaryWriter(self.args.algorithm + "logs/add_scalar1")
            self.writer2 = SummaryWriter(self.args.algorithm + "logs/add_scalar2")
        self.train_step = 0

        # create the network
        if self.args.algorithm == "CommCritic":
            self.critic_network = CommCritic(args)
            self.critic_target_network = CommCritic(args)
        elif self.args.algorithm == "base":
            self.critic_network = BaseCritic(args)
            self.critic_target_network = BaseCritic(args)

        else:
            self.critic_network = Critic(args)
            self.critic_target_network = Critic(args)

        if self.args.algorithm == "leader":
            self.actor_network = leaderActor(args)
            self.actor_target_network = leaderActor(args)
        elif self.args.algorithm == "CommCritic":
            self.actor_network = CommActor(args)
            self.actor_target_network = CommActor(args)
        elif self.args.algorithm == "base":
            self.actor_network = BaseActor(args)
            self.actor_target_network = BaseActor(args)
        else:
            self.actor_network = Actor(args)
            self.actor_target_network = Actor(args)

        self.actor_target_network.to(device)
        self.critic_target_network.to(device)
        self.actor_network.to(device)
        self.critic_network.to(device)

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr = self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr = self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.ag_path = self.args.save_dir + '/' + self.args.algorithm
        if not os.path.exists(self.ag_path):
            os.mkdir(self.ag_path)
        # path to save the model
        self.model_path = self.ag_path + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if self.args.evaluate == True:
            self.model_path = self.model_path + '/' + 'group_train'
        else:
            self.model_path = self.model_path + '/' + 'group_train%d' % self.args.agents_num
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        # 加载模型

        if os.path.exists(self.model_path + '/1_actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/1_actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/1_critic_params.pkl'))

            print('Agent  successfully loaded actor_network: {}'.format(self.model_path + '/1_actor_params.pkl'))
            print('Agent  successfully loaded critic_network: {}'.format(self.model_path + '/1_critic_params.pkl'))

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # soft update

    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        # update the network

    def train(self, buffer_batch):
        O, U, R, Next_O, Matrix, Next_Matrix, D = buffer_batch
        o = torch.Tensor(O).cuda()
        matrix = torch.Tensor(Matrix).cuda()
        o_next = torch.Tensor(Next_O).cuda()
        matrix_next = torch.Tensor(Next_Matrix).cuda()
        r = torch.Tensor(R).cuda()
        u = torch.Tensor(U).cuda()

        # calculate the target Q value function
        with torch.no_grad():
            u_next = self.actor_target_network(o_next, matrix_next)

            q_next = self.critic_target_network(o_next, u_next, matrix_next).detach()

            target_q = (r.unsqueeze(2) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_network(o, u, matrix)
        critic_loss = (target_q - q_value).pow(2).mean()
        # critic_loss=torch.nn.MSELoss()(q_value,target_q)  # bellman equation

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u = self.actor_network(o, matrix)
        actor_loss = - self.critic_network(o, u, matrix).mean()
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()

        if self.train_step % 100 == 0:
            self.writer1.add_scalar('agent0_critic_loss', critic_loss, self.train_step)
            self.writer2.add_scalar('agent0_actor_loss', actor_loss, self.train_step)
        # writer2.close()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)

        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.actor_network.state_dict(), self.model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(), self.model_path + '/' + num + '_critic_params.pkl')

