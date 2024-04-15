import numpy as np
class ReplayBuffer(object):

	def __init__(self, args):
		self.obs_space=args.obs_shape[0]
		self.action_shape=args.action_shape[0]
		self.buffer_size = args.buffer_size
		self.n_ant =args.n_agents
		self.pointer = 0
		self.current_size = 0
		self.len=0
		self.actions = np.zeros((self.buffer_size,self.n_ant,self.action_shape))
		self.rewards = np.zeros((self.buffer_size,self.n_ant))
		self.dones = np.zeros((self.buffer_size,self.n_ant))
		self.obs = np.zeros((self.buffer_size,self.n_ant,self.obs_space))
		self.next_obs = np.zeros((self.buffer_size,self.n_ant,self.obs_space))
		self.matrix = np.zeros((self.buffer_size,self.n_ant,self.n_ant))
		self.next_matrix = np.zeros((self.buffer_size,self.n_ant,self.n_ant))

	def getBatch(self, batch_size):

		index = np.random.choice(self.current_size, batch_size, replace=False)
		return self.obs[index], self.actions[index], self.rewards[index], self.next_obs[index], self.matrix[index], self.next_matrix[index], self.dones[index]

	def add(self, obs, action, reward, next_obs, matrix, next_matrix, done):

		self.obs[self.pointer] = obs
		self.actions[self.pointer] = action
		self.rewards[self.pointer] = reward
		self.next_obs[self.pointer] = next_obs
		self.matrix[self.pointer] = matrix
		self.next_matrix[self.pointer] = next_matrix
		self.dones[self.pointer] = done
		self.pointer = (self.pointer + 1)%self.buffer_size
		self.current_size = min(self.current_size + 1, self.buffer_size)
