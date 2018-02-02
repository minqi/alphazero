import numpy as np

from MCTS import MCTS

class Gym():

	def __init__(self, task, args):
		self.task = task
		self.args = args
		self.mcts = None

	def _create_contender(self, agent):
		f = agent.f.__class__(self.task)
		return agent.__class__(agent.task, f, self.args)

	def _generate_episode_data(self):
		episode_data = []
		s = self.task.get_initial_state()
		current_agent = 1
		while True:
			# plan a move using mcts
			p = self.mcts.get_planned_policy(s)
			episode_data.append((s, p, current_agent))

			# transition to next state
			a = np.random.choice(len(p), p=p)
			s, r, current_agent = self.task.get_next_state(s, a, current_agent)

			# check if task is over 
			is_complete = self.task.is_complete(s, current_agent)
			if is_complete:
				return [(d[0], d[1], r * (-1 ** (current_agent == d[2]))) for d in episode_data]

	def train(self, agent):
		args = self.args
		for i in range(args.numIterations):
			print 'Training iteration %s/%s' % (i + 1, args.numIterations) 

			agent.mcts = MCTS(self.task, agent.f, args)
			self.mcts = agent.mcts
			train_data = []

			for _ in range(args.numEpisodes):
				print 'Generating episode %s/%s' % (_ + 1, args.numEpisodes)
				train_data += self._generate_episode_data()

			# train a contender on new episode data
			contender = self._create_contender(agent)
			contender.f.train(train_data)
			win_ratios = self.contest(args.evalNumEpisodes, agent, contender)

			if (win_ratios[-1] >= args.evalWinRatio):
				agent = contender

		return agent

	def contest(self, num_matches, *agents):
		# have agents compete in task + return winner
		num_agents = len(agents)
		wins = np.zeros(num_agents + 1) # 0-index reserved for draws
		for i in range(num_matches):
			print 'Evaluating new agent, game %s/%s' % (i + 1, num_matches)
			s = self.task.get_initial_state()
			current_agent = 1
			while not self.task.is_complete(s, current_agent):
				a = agents[current_agent - 1].get_action(s)
				s, r, current_agent = self.task.get_next_state(s, a, current_agent)
				print self.task.get_canonical_form(s, 1)
				# print '-----------------'
			winner = self.task.get_winner(s)
			wins[winner] += 1
			print '====================='
			print 'winner is', winner 
			print '====================='

		return wins/float(num_matches)


