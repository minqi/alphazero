import math
import numpy as np

class MCTS():
	"""
	Monte-Carlo Tree Search with Upper Confidence Bound (UCT) estimates
	"""
	def __init__(self, task, f, args):
		self.task = task 				# Task to perform
		self.f = f 						# (Q, V) approximator to guide search

		self.args = args

		# -- Tree state --
		self.A_s = {}	    			# Valid actions from s
		self.T_s = {} 					# Terminal states --> reward
		self.N_sa = {}					# Visit counts for (s, a)
		self.N_s = {} 	    			# Visit counts for s
		self.P_s = {}					# Predicted Q(s,a) by approximator
		self.Q = {} 					# Policy value function, mapping (s, a) -> v

	def get_planned_policy(self, s, temp=1.0):
		"""
		Runs simulations of MCTS to return a
		PMF over valid actions proportional to visit counts for (s, a)
		"""
		temp = self.args.get('MCTSTemperature', temp)

		for _ in range(self.args.numSimMCTS):
			self._search(None, None, s, self.args.initAgent)

		s_hash = self.task.get_state_hash(s)
		exp_counts = [self.N_sa.get((s_hash,a), 0) ** 1./temp for a in range(self.task.get_num_actions())]
		total = sum(exp_counts)
		p = [n/float(total) for n in exp_counts]
		return p

	def _search(self, prev_s, a, s, agent):
		s_hash = self.task.get_state_hash(s)
		prev_s_hash = self.task.get_state_hash(prev_s)

		# Base cases:
		# Terminal state
		if s_hash not in self.T_s:
			if self.task.is_complete(s, agent):
				# Terminal state
				self.T_s[s_hash] = self.task.get_transition_reward(prev_s, a, s)

		if s_hash in self.T_s:
			return  self._get_canonicalized_value(self.T_s[s_hash], agent)

		# New unexplored state (leaf node)
		if s_hash not in self.P_s:
			self.P_s[s_hash], v = self._get_predicted_policy(s, agent)
			self.A_s[s_hash] = self.task.get_valid_actions(s, agent)
			self.N_s[s_hash] = 0

			return  self._get_canonicalized_value(v, agent)

		# 1. Select
		self.N_s[s_hash] += 1

		# 2. Evaluate
		best_a = self._get_max_ucb_action(s)

		# 3. Expand
		next_s, r, next_agent = self.task.get_next_state(s, best_a, agent)
		next_s = self.task.get_canonical_form(next_s, next_agent)
		v = self._search(s, a, next_s, next_agent)

		# 4. Propagate
		self._propagate_value(s, best_a, v)
		
		return self._get_canonicalized_value(v, agent)

	def _get_predicted_policy(self, s, agent):
		p, v = self.f.predict(s)
		p *= self.task.get_valid_actions(s, agent)
		p /= np.sum(p)

		return p, v

	def _get_max_ucb_action(self, s):
		s_hash = self.task.get_state_hash(s)
		best_score = -float('inf')
		best_a = 0
		for a in range(len(self.A_s[s_hash])):
			if self.A_s[s_hash][a]:
				q = self.Q.get((s_hash,a), 0)
				ucb = self.args.cpuct * self.P_s[s_hash][a] * math.sqrt(self.N_s[s_hash]) / (self.N_s.get((s_hash,a), 0) + 1)
				score = q + ucb

				if score > best_score:
					best_score = score
					best_a = a

		return best_a

	def _propagate_value(self, s, a, v):
		s_hash = self.task.get_state_hash(s)
		n_sa = self.N_sa.get((s_hash, a), 0)
		q_sa = self.Q.get((s_hash,a), 0)
		self.Q[(s_hash,a)] = (n_sa * q_sa + v) / (n_sa + 1)
		self.N_sa[(s_hash,a)] = n_sa + 1

	def _get_canonicalized_value(self, v, agent):
		return v * (-1 ** (agent != self.args.initAgent))

