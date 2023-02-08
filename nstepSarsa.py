"""
0 left
1 down
2 right
3 up
"""
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', is_slippery=False)
qValue = np.zeros([env.observation_space.n, env.action_space.n])
policy = np.full([env.observation_space.n, env.action_space.n], 0.25)
numEpisodes = 1500
stepSizeParameter = 0.5
discountFactor = 0.9
n = 3
rewardArray = np.zeros([n, 1]).reshape(-1)
actionArray = np.zeros([n, 1]).reshape(-1)
stateArray = np.zeros([n, 1]).reshape(-1)


def epsilon_greedy(Q, s, epsilon):
    if np.random.uniform() < epsilon:
        return np.random.choice(env.action_space.n)
    else:
        return np.random.choice(np.argwhere(Q[s] == np.amax(Q[s])).reshape(-1))


t = 0
for episode in range(numEpisodes):
    env.reset()
    state = 0
    T = 100000

    # print('episode', episode)
    while t < T:
        stateArray[np.mod(t, n)] = int(state)
        action = epsilon_greedy(qValue, state, 0.1)
        actionArray[np.mod(t, n)] = int(action)
        next_state, reward, done, truncated, info = env.step(action)
        # print('t: ', t, T, 'stateVisited: ', state, 'action: ', action, 'nextState: ', next_state)
        rewardArray[np.mod(t, n)] = reward
        state = next_state
        if done:
            T = t + 1
            # print('done', T)

        else:
            newAction = epsilon_greedy(qValue, next_state, 0.1)
            # print('nextAction: ', newAction)

        timeUpdate = t - n + 1
        # print('timeUpdate: ', timeUpdate)

        if timeUpdate >= 0:
            m = min(timeUpdate + n, T)
            returns = 0
            for k in range(timeUpdate + 1, m + 1):
                # print('k: ', k, 'm: ', m, np.mod(k + n - 1, n))
                returns = returns + (discountFactor ** (k - timeUpdate - 1)) * rewardArray[np.mod(k + n - 1, n)]

            # print(timeUpdate + n, T)
            if (timeUpdate + n) < T:
                # print('intime')
                returns += (discountFactor ** n) * qValue[next_state, newAction]
                # print('actionArray[timeUpdate]', int(actionArray[np.mod(timeUpdate, n)]))

            # print('state and action being updated: ', np.mod(timeUpdate, n), stateArray,
            #       stateArray[np.mod(timeUpdate, n)], actionArray, actionArray[np.mod(timeUpdate, n)])
            qValue[int(stateArray[np.mod(timeUpdate, n)]), int(actionArray[np.mod(timeUpdate, n)])] += \
                stepSizeParameter * (returns - qValue[
                    int(stateArray[np.mod(timeUpdate, n)]), int(actionArray[np.mod(timeUpdate, n)])])

        t += 1

        if timeUpdate == T - 1:
            print('hi')
            break

print('qValue', qValue)

# OBS: nstepSARSA works good for discounting
