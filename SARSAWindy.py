import gym
import numpy as np
from create_02 import CustomEnvironment

env = CustomEnvironment(render_mode='rgb_array')

# SARSA implementation in example 6.5

qValue = np.zeros([70, env.action_space.n])
policy = np.zeros([69, env.action_space.n])
numEpisodes = 200
stepSizeParameter = 0.5
discountFactor = 1


def epsilon_greedy(Q, s, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(env.action_space.n)
    else:
        return np.random.choice(np.argwhere(Q[s] == np.amax(Q[s])).reshape(-1))


for episode in range(numEpisodes):
    done = False
    state = 0
    env.reset()
    action = epsilon_greedy(qValue, state, 0.1)
    while not done:
        next_state, reward, done, truncated, info = env.step(action)
        nxt = next_state
        next_state = next_state[1]*10 + next_state[0]
        # print(nxt, next_state)
        nextAction = epsilon_greedy(qValue, next_state, 0.1)
        qValue[state, action] += stepSizeParameter*(reward + discountFactor*qValue[next_state][nextAction]
                                                    - qValue[state][action])
        state = next_state
        action = nextAction
    print('terminated at state',episode,  nxt)

for i in range(qValue.shape[0]):
    print(f'index [{int(i/10)},{np.mod(i, 10)}] qvalue {qValue[i]}')
# print(f'qTable after {numEpisodes} episodes: \n {qValue}')

