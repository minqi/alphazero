import numpy as np

from Agent import Agent

class RandomAgent(Agent):

	def __init__(self, task):
		self.task = task

	def get_action(self, s):
		mask = self.task.get_valid_actions(s)
		valid_actions = list(filter(lambda a: mask[a] > 0, range(len(mask))))
		return valid_actions[np.random.randint(len(valid_actions))]
