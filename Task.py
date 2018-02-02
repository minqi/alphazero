class Task(object):
	# Override in subclass

	def __init__(self, init_agent=1):
		self.init_agent = init_agent

	def get_initial_state(self, agent):
		pass

	def get_canonical_form(self, s, agent):
		pass

	def get_symmetries(self, s):
		pass

	def get_next_state(self, s, a, agent):
		pass

	def get_transition_reward(self, s, a, next_s):
		pass

	def get_state_shape(self):
		pass

	def get_num_agents(self):
		pass

	def get_num_actions(self):
		pass

	def get_valid_actions(self, s, agent):
		pass

	def has_valid_actions(self, s, agent):
		pass

	def is_complete(self, s, agent):
		pass

	def get_winner(self, s):
		pass

	def get_state_hash(self, s):
		pass