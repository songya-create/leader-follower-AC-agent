import math, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)


class ATT(nn.Module):
    def __init__(self, din):
        super(ATT, self).__init__()
        self.fc1 = nn.Linear(din, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.sigmoid(self.fc3(y))
        return y


class Encoder(nn.Module):
    def __init__(self, din=32, hidden_dim=128):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(din, hidden_dim)

    def forward(self, x):
        embedding = F.relu(self.fc(x))
        return embedding


##与不加人注意力机制对比
class AttModel(nn.Module):
    def __init__(self, din, hidden_dim, dout):
        super(AttModel, self).__init__()

        self.fcv = nn.Linear(din, hidden_dim)
        self.fck = nn.Linear(din, hidden_dim)
        self.fcq = nn.Linear(din, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, dout)

    def forward(self, x, mask):
        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        k = F.relu(self.fck(x)).permute(0, 2, 1)  ##自注意力机制
        att = F.softmax(torch.mul(torch.bmm(q / (64 * (6) ** 2), k), mask) - 9e15 * (1 - mask), dim = 2)

        out0 = torch.bmm(att, v)  # 计算两个矩阵相乘
        # out1 = torch.add(out0,v)#记得删除
        out = torch.relu(out0)
        return out


class Q_Net(nn.Module):
    def __init__(self, hidden_dim, dout):
        super(Q_Net, self).__init__()
        self.fc = nn.Linear(hidden_dim, dout)

    def forward(self, x):
        q = self.fc(x)
        return q


'''
class Actor(nn.Module):
    def __init__(self,args):
        super(Actor, self).__init__()
        num_inputs=args.obs_shape[0]
        hidden_dim=256
        num_actions=args.action_shape[0]

        self.encoder = Encoder(6, hidden_dim)
        self.hob0= nn.Linear(num_inputs,  hidden_dim)

        self.att_1 = AttModel(hidden_dim, hidden_dim, hidden_dim)
        self.att_2 = AttModel(hidden_dim,hidden_dim,hidden_dim)
        self.fout1= nn.Linear(hidden_dim+hidden_dim,hidden_dim)
        self.fout = nn.Linear(hidden_dim, num_actions)

    def forward(self, x, mask):
        h1 = self.encoder(x[:,:,0:6])  # 输入经过一层mlp或者cnn,x格式是输入的(batch,agent数,agent观测信息）
        hloacl0 = torch.relu(self.hob0(x))

        h2 = self.att_1(h1, mask)  # 3,h1size:[1, 100, 64]
        h3 =self.att_2(h2, mask)

        hout=torch.relu(self.fout1(torch.cat([h3,hloacl0],dim=2)))
        action=torch.tanh(self.fout(hout)) ##这里就可以直接输出动作了
        return action


class Critic(nn.Module):
    def __init__(self,args):
        super(Critic, self).__init__()
        num_inputs=args.obs_shape[0]+args.action_shape[0]
        hidden_dim=128
        self.mas = torch.ones((args.batch_size,args.n_agents, args.n_agents))
        num_actions=args.action_shape[0]

        self.encoder = Encoder(num_inputs,hidden_dim)
        self.att_1 = AttModel(hidden_dim, hidden_dim, hidden_dim)
        self.att_2 = AttModel(hidden_dim, hidden_dim, hidden_dim)
        self.fout1=nn.Linear(hidden_dim, 64)
        self.fout= nn.Linear(64, 1)

    def forward(self, ob, ac,mask):
        mas=self.mas.cuda()
        x = torch.cat([ob, ac], dim=2)
        h1 = self.encoder(x)  # 输入经过一层mlp或者cnn,x格式是输入的(batch,agent数,agent观测信息）
        h2 = self.att_1(h1, mas)  # 3,h1size:[1, 100, 64]
        h3 = self.att_2(h2, mas)
        hout = torch.relu(self.fout1(h3))

        q=self.fout(hout) ##这里就可以直接输出动作了
        return q
'''


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        num_inputs = 6
        hidden_dim = 256
        num_actions = args.action_shape[0]

        self.encoder = Encoder(num_inputs, hidden_dim)
        self.att_1 = AttModel(hidden_dim, hidden_dim, hidden_dim)
        self.att_2 = AttModel(hidden_dim, hidden_dim, hidden_dim)
        self.fout1 = nn.Linear(hidden_dim, hidden_dim)
        self.fout = nn.Linear(hidden_dim, num_actions)

    def forward(self, x, mask):
        h1 = self.encoder(x[:, :, 0:6])  # 输入经过一层mlp或者cnn,x格式是输入的(batch,agent数,agent观测信息）
        h2 = self.att_1(h1, mask)  # 3,h1size:[1, 100, 64]
        h3 = self.att_2(h2, mask)

        hout = torch.relu(self.fout1(h3))
        action = torch.tanh(self.fout(hout))  ##这里就可以直接输出动作了
        return action


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        num_inputs = 6 + args.action_shape[0]
        hidden_dim = 256
        self.mas = torch.ones((args.batch_size, args.n_agents, args.n_agents))
        num_actions = args.action_shape[0]

        self.encoder = Encoder(num_inputs, hidden_dim)
        self.att_1 = AttModel(hidden_dim, hidden_dim, hidden_dim)
        self.att_2 = AttModel(hidden_dim, hidden_dim, hidden_dim)
        self.fout1 = nn.Linear(hidden_dim, 64)
        self.fout = nn.Linear(64, 1)

    def forward(self, ob, ac, mask):
        mas = self.mas.cuda()
        x = torch.cat([ob[:, :, 0:6], ac], dim = 2)
        h1 = self.encoder(x)  # 输入经过一层mlp或者cnn,x格式是输入的(batch,agent数,agent观测信息）
        h2 = self.att_1(h1, mas)  # 3,h1size:[1, 100, 64]
        h3 = self.att_2(h2, mas)
        hout = torch.relu(self.fout1(h3))

        q = self.fout(hout)  ##这里就可以直接输出动作了
        return q


class CommCritic(nn.Module):
    def __init__(self, args):
        super(CommCritic, self).__init__()
        num_inputs = args.obs_shape[0] + args.action_shape[0]
        hidden_dim = 128
        self.args = args
        self.encoder = Encoder(num_inputs, hidden_dim)
        self.h1 = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.h2 = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.fout = nn.Linear(hidden_dim, 1)

    def forward(self, ob, ac, mask):
        x = torch.cat([ob, ac], dim = 2)
        x1 = self.encoder(x)  # 输入经过一层mlp或者cnn,x格式是输入的(batch,agent数,agent观测信息）
        ad1 = (torch.sum(x1, dim = 1)).unsqueeze(1).repeat(1, self.args.agents_num, 1)
        ad1 = ad1 / self.args.agents_num
        h1 = torch.relu(self.h1(torch.cat([ad1, x1], dim = 2)))  # 3,h1size:[1, 100, 64]
        ad2 = (torch.sum(h1, dim = 1)).unsqueeze(1).repeat(1, self.args.agents_num, 1)
        ad2 = ad2 / self.args.agents_num
        h2 = torch.relu(self.h2(torch.cat([ad2, h1], dim = 2)))
        q = self.fout(h2)
        return q


class CommActor(nn.Module):
    def __init__(self, args):
        super(CommActor, self).__init__()
        num_inputs = args.obs_shape[0]
        hidden_dim = 256
        self.args = args
        num_actions = args.action_shape[0]
        self.encoder = Encoder(num_inputs, hidden_dim)
        self.h1 = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.h2 = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.fout = nn.Linear(hidden_dim, num_actions)

    def forward(self, ob, mask):
        x1 = self.encoder(ob)  # 输入经过一层mlp或者cnn,x格式是输入的(batch,agent数,agent观测信息）
        ad1 = (torch.sum(x1, dim = 1)).unsqueeze(1).repeat(1, self.args.agents_num, 1)
        ad1 = ad1 / self.args.agents_num
        h1 = torch.relu(self.h1(torch.cat([ad1, x1], dim = 2)))  # 3,h1size:[1, 100, 64]
        ad2 = (torch.sum(h1, dim = 1)).unsqueeze(1).repeat(1, self.args.agents_num, 1)
        ad2 = ad2 / self.args.agents_num
        h2 = torch.relu(self.h2(torch.cat([ad2, h1], dim = 2)))
        actions = torch.tanh(self.fout(h2))
        return actions


class leaderActor(nn.Module):
    def __init__(self, args):
        super(leaderActor, self).__init__()
        num_inputs = args.obs_shape[0]
        hidden_dim = 256
        num_actions = args.action_shape[0]

        self.encoder = Encoder(num_inputs, hidden_dim)
        self.hob1 = nn.Linear(num_inputs, 64)
        self.att_1 = AttModel(hidden_dim, hidden_dim, hidden_dim)
        self.att_2 = AttModel(hidden_dim, hidden_dim, hidden_dim)
        self.fenv = nn.Linear(320, hidden_dim)
        self.fout0 = nn.Linear(hidden_dim, 64)
        self.fout = nn.Linear(64, num_actions)
        self.lfout0 = nn.Linear(hidden_dim, 64)
        self.lfout = nn.Linear(64, num_actions)

    def forward(self, x, mask):
        h1 = self.encoder(x)  # 输入经过一层mlp或者cnn,x格式是输入的(batch,agent数,agent观测信息）
        hloacl1 = self.hob1(x)
        h2 = self.att_1(h1, mask)  # 3,h1size:[1, 100, 64]
        h3 = self.att_2(h2, mask)
        henv = torch.relu(self.fenv(torch.cat([h3, hloacl1], dim = 2)))

        # 输出阶段
        lout0 = torch.relu(self.lfout0(henv[:, 0, :]))
        hout0 = torch.relu(self.fout0(henv[:, 1:, :]))
        lout = torch.tanh(self.lfout(lout0))
        hout = torch.tanh(self.fout(hout0))

        action = torch.cat([lout.unsqueeze(1), hout], dim = 1)
        return action


class BaseCritic(nn.Module):
    def __init__(self, args):
        super(BaseCritic, self).__init__()
        num_inputs = args.obs_shape[0] + args.action_shape[0]
        hidden_dim = 128
        self.args = args
        self.encoder = Encoder(num_inputs, hidden_dim)
        self.h1 = nn.Linear(hidden_dim, hidden_dim)
        self.h2 = nn.Linear(hidden_dim, hidden_dim)
        self.fout = nn.Linear(hidden_dim, 1)

    def forward(self, ob, ac, mask):
        x = torch.cat([ob, ac], dim = 2)
        x1 = self.encoder(x)  # 输入经过一层mlp或者cnn,x格式是输入的(batch,agent数,agent观测信息）

        h1 = torch.relu(self.h1(x1))  # 3,h1size:[1, 100, 64]
        h2 = torch.relu(self.h2(h1))
        q = self.fout(h2)
        return q


class BaseActor(nn.Module):
    def __init__(self, args):
        super(BaseActor, self).__init__()
        num_inputs = args.obs_shape[0]
        hidden_dim = 256
        self.args = args
        num_actions = args.action_shape[0]
        self.encoder = Encoder(num_inputs, hidden_dim)
        self.h1 = nn.Linear(hidden_dim, hidden_dim)
        self.h2 = nn.Linear(hidden_dim, hidden_dim)
        self.fout = nn.Linear(hidden_dim, num_actions)

    def forward(self, ob, mask):
        x1 = self.encoder(ob)  # 输入经过一层mlp或者cnn,x格式是输入的(batch,agent数,agent观测信息）

        h1 = torch.relu(self.h1(x1))  # 3,h1size:[1, 100, 64]

        h2 = torch.relu(self.h2(h1))
        actions = torch.tanh(self.fout(h2))
        return actions
