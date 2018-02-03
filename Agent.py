import numpy as np

from MCTS import MCTS

class Agent():

	def __init__(self, task, f, args):
		self.task = task
		self.f = f
		self.mcts = MCTS(task, f, args)

	def get_action(self, s):
		p = self.mcts.get_planned_policy(s)
		return np.random.choice(len(p), p=p)

	def get_random_action(self, s):
		mask = self.task.get_valid_actions(s)
		valid_actions = list(filter(lambda a: mask[a] > 0, range(len(mask))))
		return valid_actions[np.random.randint(len(valid_actions))]
