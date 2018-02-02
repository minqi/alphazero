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