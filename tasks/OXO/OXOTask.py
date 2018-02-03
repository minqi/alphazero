import numpy as np

from Task import Task

class OXOTask(Task):

	state_shape = (3, 3)
	num_agents = 2

	def get_initial_state(self):
		return np.zeros(shape=OXOTask.state_shape).astype('int16')

	def get_canonical_form(self, s, agent):
		return np.array([((x - agent) % OXOTask.num_agents) + 1 if x > 0 else x for x in s.ravel()]).reshape(s.shape)

	def get_symmetries(self, s, p):
		symmetries = []

		s_copy = np.copy(s).reshape(OXOTask.state_shape)
		p_copy = np.copy(p).reshape(b.shape)

		for i in range(4):
			s_rotated = np.rot90(s_copy, i)
			p_rotated = np.rot90(p_copy, i)

			symmetries.append((rotated, p_rotated.flatten())) # rotations
			symmetries.append((np.fliplr(s_rotated), np.fliplr(p_rotated).flatten())) # reflections

		return symmetries

	def get_next_agent(self, agent):
		return max((agent + 1) % (OXOTask.num_agents + 1), 1)

	def get_next_state(self, s, a, agent):
		assert s.ravel()[a] == 0
		next_s = np.copy(s)
		next_s.ravel()[a] = agent
		next_agent = self.get_next_agent(agent)
		r = self.get_transition_reward(s, a, next_s)
		return next_s, r, next_agent

	def get_transition_reward(self, s, a, next_s):
		reward = 1
		winner = self.get_winner(next_s)
		if winner > 0:
			return reward * (-1 ** (self.init_agent != winner))

		return 0 

	def get_state_shape(self):
		return OXOTask.state_shape

	def get_num_agents(self):
		return OXOTask.num_agents

	def get_num_actions(self):
		return OXOTask.state_shape[0] ** 2

	def get_valid_actions(self, s, agent=None):
		return [int(x == 0) for x in s.ravel()]

	def has_valid_actions(self, s, agent):
		return sum([int(x == 0) for x in s.ravel()]) > 0

	def is_complete(self, s, agent):
		return self.get_winner(s) > 0 or sum([int(x == 0) for x in s.ravel()]) == 0

	def get_winner(self, s):
		# check rows
		init_agent = self.init_agent
		complete = False
		reward = 1
		for row in s:
			complete = np.array_equal(row[1:], row[:-1]) and row[0] > 0
			if complete: 
				return row[0]

		# check columns
		for row in np.rot90(s):
			complete = np.array_equal(row[1:],row[:-1]) and row[0] > 0
			if complete: 
				return row[0]

		# check diagonals
		n = OXOTask.state_shape[0]
		lr_diag = s.ravel()[[i * n + i for i in range(n)]]
		complete = np.array_equal(lr_diag[1:], lr_diag[:-1]) and lr_diag[0] > 0
		if complete: 
			return lr_diag[0]

		rl_diag = s.ravel()[[(i + 1) * n - (i + 1) for i in range(n)]]
		complete = np.array_equal(rl_diag[1:], rl_diag[:-1]) and rl_diag[0] > 0
		if complete: 
			return rl_diag[0]

		return 0

	def get_state_hash(self, s):
		if s is None:
			return None
		return np.array_str(s)

