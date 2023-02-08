"""
Value evaluation using DP of Frozen Lake env
"""

import matplotlib.pyplot as plt
import gym
import numpy as np


env = gym.make('FrozenLake-v1', render_mode="rgb_array", is_slippery=True)
env.reset()

nS = env.observation_space.n
nA = env.action_space.n

policy = np.full(nS*nA, 0.25).reshape(nS, nA)

# policy is the probability of taking certain action when in a certain state
V = np.zeros(16)  # Initializing the value of all states to be 0

def policyEvaluation(policy, env, discountFactor = 1, theta = 1e-9):
    V = np.zeros(16)
    for k in range(int(66)):

        delta= 0
        for state in range(nS):
            # for a particular state

            value = 0
            for action in range(nA):
                actionProbability = policy[state][action]
                innerSum = 0
                for stateTransitionProbability, nextState, rd, terminated in env.P[state][action]:
                    previousStateValue = V[nextState]  # Previous Value of the new state it might end up in after taking action
                    innerSum = innerSum + stateTransitionProbability*previousStateValue

                reward = env.P[state][action][0][2]

                value = value + actionProbability*(reward + innerSum)
            delta = max(delta, np.abs(V[state] - value))
            V[state] = value

        if delta < theta:
            print(f'Policy evaluated in {k} iterations.')
            return V


print(policyEvaluation(policy, env))
