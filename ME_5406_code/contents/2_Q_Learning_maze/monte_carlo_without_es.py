import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from maze_env import Maze
import seaborn as sns


# play a game
# @policyPlayerFn: specify policy for player
# @initialState: [whether player has a usable Ace, sum of player's cards, one card of dealer]
# @initialAction: the initial action
def play(policyPlayerFn, initialState=None, initialAction=None):
	# player status

	# sum of player
	playerSum = 0

	# trajectory of player
	playerTrajectory = []

	# whether player uses Ace as 11
	usableAcePlayer = False

	# dealer status
	dealerCard1 = 0
	dealerCard2 = 0
	usableAceDealer = False

	if initialState is None:
		# generate a random initial state

		numOfAce = 0

		# initialize cards of player
		while playerSum < 12:
			# if sum of player is less than 12, always hit
			card = getCard()

			# if get an Ace, use it as 11
			if card == 1:
				numOfAce += 1
				card = 11
				usableAcePlayer = True
			playerSum += card

		# if player's sum is larger than 21, he must hold at least one Ace, two Aces are possible
		if playerSum > 21:
			# use the Ace as 1 rather than 11
			playerSum -= 10

			# if the player only has one Ace, then he doesn't have usable Ace any more
			if numOfAce == 1:
				usableAcePlayer = False

		# initialize cards of dealer, suppose dealer will show the first card he gets
		dealerCard1 = getCard()
		dealerCard2 = getCard()

	else:
		# use specified initial state
		usableAcePlayer = initialState[0]
		playerSum = initialState[1]
		dealerCard1 = initialState[2]
		dealerCard2 = getCard()

	# initial state of the game
	state = [usableAcePlayer, playerSum, dealerCard1]

	# initialize dealer's sum
	dealerSum = 0
	if dealerCard1 == 1 and dealerCard2 != 1:
		dealerSum += 11 + dealerCard2
		usableAceDealer = True
	elif dealerCard1 != 1 and dealerCard2 == 1:
		dealerSum += dealerCard1 + 11
		usableAceDealer = True
	elif dealerCard1 == 1 and dealerCard2 == 1:
		dealerSum += 1 + 11
		usableAceDealer = True
	else:
		dealerSum += dealerCard1 + dealerCard2

	# game starts!

	# player's turn
	while True:
		if initialAction is not None:
			action = initialAction
			initialAction = None
		else:
			# get action based on current sum
			action = policyPlayerFn(usableAcePlayer, playerSum, dealerCard1)

		# track player's trajectory for importance sampling
		playerTrajectory.append([(usableAcePlayer, playerSum, dealerCard1), action])

		if action == ACTION_STAND:
			break
		# if hit, get new card
		playerSum += getCard()

		# player busts
		if playerSum > 21:
			# if player has a usable Ace, use it as 1 to avoid busting and continue
			if usableAcePlayer == True:
				playerSum -= 10
				usableAcePlayer = False
			else:
				# otherwise player loses
				return state, -1, playerTrajectory

	# dealer's turn
	while True:
		# get action based on current sum
		action = policyDealer[dealerSum]
		if action == ACTION_STAND:
			break
		# if hit, get a new card
		new_card = getCard()
		if new_card == 1 and dealerSum + 11 < 21:
			dealerSum += 11
			usableAceDealer = True
		else:
			dealerSum += new_card
		# dealer busts
		if dealerSum > 21:
			if usableAceDealer == True:
				# if dealer has a usable Ace, use it as 1 to avoid busting and continue
				dealerSum -= 10
				usableAceDealer = False
			else:
				# otherwise dealer loses
				return state, 1, playerTrajectory

	# compare the sum between player and dealer
	if playerSum > dealerSum:
		return state, 1, playerTrajectory
	elif playerSum == dealerSum:
		return state, 0, playerTrajectory
	else:
		return state, -1, playerTrajectory


def sample_one_trajectory(env, behaviorPolicy, stateActionValues, stateActionPairCount):
	# for each episode, use a randomly initialized state and action
	initialState = np.random.choice(list(range(env.n_states)))
	initialAction = np.random.choice(list(range(env.n_actions)))
	print("initstate :", initialState)
	print("initaction :", initialAction)

	# trajectory of player
	playerTrajectory = []

	obs, state = env.reset()
	action = initialAction
	done = False
	while done is False:
		observation_, state_, reward, done = env.step(action)
		playerTrajectory.append([state, action, reward])
		state = state_
		action = behaviorPolicy(state, stateActionValues, stateActionPairCount)

	return playerTrajectory


def monteCarloNoES(nEpisodes=10):
	# set up environment
	env = Maze()

	# (playerSum, dealerCard, usableAce, action)
	stateActionValues = np.zeros((16, 4))

	# initialze counts to 1 to avoid division by 0
	stateActionPairCount = np.ones((16, 4))

	# define greedy policy
	def behaviorPolicy(state, stateActionValues, stateActionPairCount):
		values_ = stateActionValues[state, :] / stateActionPairCount[state, :]

		return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

	# play for several episodes
	for episode in range(nEpisodes):
		if episode % 10 == 0:
			print('episode:', episode)

		trajectory = sample_one_trajectory(env, behaviorPolicy, stateActionValues, stateActionPairCount)

		print("trajectory :::", trajectory)
		for state, action, reward in reversed(trajectory):

			# update values of state-action pairs
			stateActionValues[state, action] += reward
			stateActionPairCount[state, action] += 1

	return stateActionValues / stateActionPairCount


if __name__ == "__main__":
	s_a_distribution = monteCarloNoES(nEpisodes=10)
	print("state_value ::", s_a_distribution)
	s_distribution = np.sum(s_a_distribution, axis=1)
	fig = plt.figure(figsize=(12, 6))
	plt.subplot(121)
	ax_1 = sns.heatmap(s_a_distribution, cbar=True)
	print("state_value ::", s_distribution)
	plt.subplot(122)
	ax_2 = sns.heatmap(s_distribution.reshape((4, 4)), cbar=True)
	# heatmap = plt.pcolor(value_distribution, cmap='RdBu')
	plt.show()
