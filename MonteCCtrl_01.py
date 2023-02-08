import gym
import matplotlib.pyplot as plt
import numpy as np

"""
0: LEFT

1: DOWN

2: RIGHT

3: UP
"""

env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='rgb_array')
# qValue is an array of 16*4 in this case
qValue = np.zeros([env.observation_space.n, env.action_space.n])
returns = np.zeros([env.observation_space.n, env.action_space.n])
policy = np.full([env.observation_space.n, env.action_space.n], 0.25)
arrayPolicy = np.zeros([env.observation_space.n, env.action_space.n])
numOfEpisodes = 10000


def policyCalculation(st, q, epsilon=0.9):
    nonGreedyProb = epsilon / env.action_space.n
    greedyProb = 1 - epsilon + (epsilon / env.action_space.n)
    ac = np.random.choice(np.argwhere(q[st] == np.amax(q[st])).reshape(-1))
    arrayPolicy[st][ac] = greedyProb

    # [i for i in range(env.action_space.n) if i != ac]
    for k in range(env.action_space.n):
        if k != ac:
            arrayPolicy[st][k] = nonGreedyProb

    return arrayPolicy


def epsilon_greedy(Q, s, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(env.action_space.n)
    else:
        return np.random.choice(np.argwhere(Q[s] == np.amax(Q[s])).reshape(-1))


gamma = 0.9

for episode in range(numOfEpisodes):
    G = 0
    env.reset()
    state = 0
    done = False
    record = list()

    while not done:
        action = epsilon_greedy(returns, state, 0.9)
        next_state, reward, done, truncated, info = env.step(action)
        record.append((state, action, reward))
        state = next_state

    for step in reversed(range(len(record))):
        currentState = record[step][0]
        currentAction = record[step][1]
        rewardGained = record[step][2]
        G = gamma * G + rewardGained
        returns[record[step][0]][record[step][1]] += G
        policy = policyCalculation(currentState, returns)

qValue = returns / numOfEpisodes
print(
    f'Returns after {numOfEpisodes} episodes is \n {returns} \n and qValue is \n {qValue} \n and policy is \n {policy} \n')
# plt.imshow(env.render())
# plt.show(block=False)
# plt.pause(0.4)
# plt.close()
