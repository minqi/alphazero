from Gym import Gym
from Agent import Agent
from RandomAgent import RandomAgent
from tasks.OXO.OXOTask import OXOTask
from tasks.OXO.OXONeuralNetwork import OXONeuralNetwork as OXONN
from util import DotDic

args = DotDic({
	'numIterations': 1500,
	'numEpisodes': 20,
	'updateLimit': 0.6,
	'maxExamplesLength': 10000,
	'numSimMCTS': 30,
	'MCTSTemperature': 1.2,
	'evalNumEpisodes': 30,
	'evalWinRatio': .55,
	'cpuct': 1,
	'initAgent': 1,
})

if __name__ == '__main__':
	# initialize Gym with task + neural network
	task = OXOTask()
	gym = Gym(task, args)
	agent = gym.train(Agent(task, OXONN(task), args))

	random_agent = RandomAgent(task)
	
	# benchmark agent against random agent (selects using uniform pmf over valid actions per state)
	# outputs [draw ratio, agent win ratio, random agent win ratio]
	print gym.contest(100, agent, random_agent)

	# @todo: play with human agent via command line input
