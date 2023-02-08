"""
SARSA implementation in example 6.5

0 right
1 down
2 left
3 up
"""

import numpy as np
import matplotlib.pyplot as plt
from cliffWalking import CustomEnvironment

float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})
env = CustomEnvironment(render_mode = 'rgb_array')

qValue = np.zeros([48, 4])
numEpisodes = 500
stepSizeParameter = 0.5
discountFactor = 0.9


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
        next_state = next_state[1]*12 + next_state[0]
        # print(nxt, next_state)
        nextAction = epsilon_greedy(qValue, next_state, 0.1)
        qValue[state, action] += stepSizeParameter*(reward + discountFactor*qValue[next_state][nextAction]
                                                    - qValue[state][action])
        state = next_state
        action = nextAction
    print('terminated at state',episode,  nxt)

for i in range(qValue.shape[0]):
    print(f'index [{int(i/12)},{np.mod(i, 12)}] qvalue {qValue[i]}')