import numpy as np
import torch
import os
from D_GAT_AC.D_GAT_AC import D_GAT_AC
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self,args):
        self.args = args
        self.policy = D_GAT_AC(args)

    def select_action(self, o,mask,epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, [self.args.n_agents ,self.args.action_shape[0]])
        else:
            inputs = torch.tensor(np.array([o]), dtype=torch.float32).to(device)
            input_mask=torch.tensor(np.array([mask]), dtype=torch.int).to(device)
            actionp=self.policy.actor_network(inputs,input_mask)
            pi = actionp.squeeze(0)
            # print('{} : {}'.format(self.name, pi))

            noise =torch.tensor(np.random.uniform(0.0,self.args.noise_rate,(self.args.agents_num,self.args.action_shape[0])),
                             dtype=torch.float).to(device)
            action = torch.clamp(pi +noise, -1, 1)
            u = action.cpu().numpy()
        return u.copy()

    def learn(self,buffer_batch):
        self.policy.train(buffer_batch)

