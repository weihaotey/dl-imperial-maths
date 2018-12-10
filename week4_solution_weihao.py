import numpy as np
import gym
import matplotlib.pyplot as plt
env = gym.make('FrozenLake-v0')
# alpha = 0.1 #learning rate
# gamma = 0.99 #discount factor
# total_episode = 15000
# trials = 10
# average = 0 #for computing average success rate of att trials
# decay_rate = 0.005
# max_timestep = 1000
# min_epsilon = 0.01
# max_epsilon = 1
# epsilon = 1  # initial value of epsilon
# # epsilon = 0.1 #try different constant values of epsilon
#action_size = env.action_space.n
#state_size = env.observation_space.n

def frozenSARSA(Q,total_episode,max_timestep):

    env = gym.make('FrozenLake-v0')
    alpha = 0.1  # learning rate
    gamma = 0.99  # discount factor
    average = 0  # for computing average success rate of att trials
    decay_rate = 0.005
    min_epsilon = 0.01
    max_epsilon = 1
    epsilon = 1  # initial value of epsilon
    # epsilon = 0.1 #try different constant values of epsilon
    action_size = env.action_space.n
    state_size = env.observation_space.n
    total_reward = 0  # record the total number of time it reach the goal
    # found that fluctuations occur when uses random matrix Q, but if initialise as zero matrix, found that
    # if there is no success in any round then the overall success rate will be zero, since matrix stay zero
    success_rate = np.zeros(int(total_episode/100))
    k = 0  # indices for success rate
    for i_episode in range(total_episode):
        observation = env.reset()
        action = 0  # for SARSA algorithm

        for t in range(max_timestep):
            prvobs = observation  # to record the previous observation (state)
            prvact = action #to record previous action (for SARSA algorithm)
            # epsilon-greedy
            z = np.random.choice([0, 1], p=[1 - epsilon, epsilon])
            if z == 0:  # with probability of 1- epsilon
                action = np.argmax(Q[observation, :])  # best path according to Q table
            else:  # with probability of epsilon
                action = env.action_space.sample()  # random choice

            observation, reward, done, info = env.step(action)
            total_reward += reward

            # strategy that act on the next step of Q value (SARSA algorithm)
            if t != 0:
               Q[prvobs, prvact] += alpha * (reward + gamma * (Q[observation, action]) - Q[prvobs, prvact])
            prvobs = observation  # to record the previous observation (state)

            if done:
                break

        # update for exponentially decaying epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * i_episode)
        if (i_episode + 1) % 100 == 0:
            success_rate[k] = total_reward / 100
            total_reward = 0
            k += 1
    print('success rate: ', np.mean(success_rate) * 100, 'percent, with maximum', np.max(success_rate) * 100,
          'percent, out of 100 trials')

    plt.plot(np.arange(0, total_episode / 100), success_rate)
    plt.show()
    return 0

def frozenQ(Q,total_episode,max_timestep):
    env = gym.make('FrozenLake-v0')
    alpha = 0.1  # learning rate
    gamma = 0.99  # discount factor
    average = 0  # for computing average success rate of att trials
    decay_rate = 0.005
    min_epsilon = 0.01
    max_epsilon = 1
    epsilon = 1  # initial value of epsilon
    # epsilon = 0.1 #try different constant values of epsilon
    action_size = env.action_space.n
    state_size = env.observation_space.n
    total_reward = 0  # record the total number of time it reach the goal
    # found that fluctuations occur when uses random matrix Q, but if initialise as zero matrix, found that
    # if there is no success in any round then the overall success rate will be zero, since matrix stay zero
    success_rate = np.zeros(int(total_episode/100))
    k = 0  # indices for success rate
    for i_episode in range(total_episode):
        observation = env.reset()
        action = 0  # for SARSA algorithm

        for t in range(max_timestep):
            prvobs = observation  # to record the previous observation (state)
            # epsilon-greedy
            z = np.random.choice([0, 1], p=[1 - epsilon, epsilon])
            if z == 0:  # with probability of 1- epsilon
                action = np.argmax(Q[observation, :])  # best path according to Q table
            else:  # with probability of epsilon
                action = env.action_space.sample()  # random choice

            observation, reward, done, info = env.step(action)
            total_reward += reward

            # strategy that act on maximum of current action and observation
            Q[prvobs, action] += alpha * (reward + gamma * (max(Q[observation, :])) - Q[prvobs, action])

            if done:
                break

        # update for exponentially decaying epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * i_episode)
        if (i_episode + 1) % 100 == 0:
            success_rate[k] = total_reward / 100
            total_reward = 0
            k += 1
    print('success rate: ', np.mean(success_rate) * 100, 'percent, with maximum', np.max(success_rate) * 100,
          'percent, out of 100 trials')

    plt.plot(np.arange(0, total_episode / 100), success_rate)
    plt.show()
    return 0

action_size = env.action_space.n
state_size = env.observation_space.n
trials = 10
Q = np.random.rand(state_size, action_size) * 0  # initialize Q table
episode = 15000
max_step = 1000
for j in range(trials):
     #found that fluctuations occur when uses random matrix Q, but if initialise as zero matrix, found that
     #if there is no success in any round then the overall success rate will be zero, since matrix stay zero
     frozenQ(Q,episode,max_step)
     frozenSARSA(Q,episode,max_step)

# for j in range(trials):
#     total_reward = 0 #record the total number of time it reach the goal
#     #found that fluctuations occur when uses random matrix Q, but if initialise as zero matrix, found that
#     #if there is no success in any round then the overall success rate will be zero, since matrix stay zero
#     Q = np.random.rand(state_size, action_size)*0 #initialize Q table
#     success_rate = np.zeros(150)
#     k = 0 #indices for success rate
#     for i_episode in range(total_episode):
#         observation = env.reset()
#         action = 0 #for SARSA algorithm
#
#         for t in range(max_timestep):
#             prvobs = observation  # to record the previous observation (state)
#             #prvact = action #to record previous action (for SARSA algorithm)
#             # epsilon-greedy
#             z = np.random.choice([0,1],p=[1-epsilon,epsilon])
#             if z == 0: #with probability of 1- epsilon
#                 action = np.argmax(Q[observation, :])  # best path according to Q table
#             else: #with probability of epsilon
#                 action = env.action_space.sample() #random choice
#
#             observation, reward, done, info = env.step(action)
#             total_reward += reward
#
#             #strategy that act on maximum of current action and observation
#             Q[prvobs, action] += alpha * (reward + gamma * (max(Q[observation, :])) - Q[prvobs, action])
#
#             #strategy that act on the next step of Q value (SARSA algorithm)
#             #if t != 0:
#             #    Q[prvobs, prvact] += alpha * (reward + gamma * (Q[observation, action]) - Q[prvobs, prvact])
#             #prvobs = observation  # to record the previous observation (state)
#
#             if done:
#                 #print("Episode finished after {} timesteps".format(t+1))
#                 #env.render()
#                 break
#
#         # update for exponentially decaying epsilon
#         epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * i_episode)
#         if (i_episode+1)%100 == 0:
#             success_rate[k] = total_reward/100
#             total_reward = 0
#             k += 1
#     average += np.mean(success_rate) #for average success rate
#     print('success rate: ',np.mean(success_rate)*100,'percent, with maximum', np.max(success_rate)*100,'percent, out of 100 trials')
#
#
#     plt.plot(np.arange(0, total_episode/100), success_rate)
#     plt.show()
#     #print(Q)
# print('total success rate: ',average/trials*100,'percent \n')



#import gym
#env = gym.make('CartPole-v0')
#print(env.action_space) #allows a fixed range of non-negative numbers, so in this case valid actions are either 0 or 1.
#> Discrete(2)
#print(env.observation_space) #represents an n-dimensional box, so valid observations will be an array of 4 numbers.
#> Box(4,)

#print(env.observation_space.high)
#> array([ 2.4       ,         inf,  0.20943951,         inf])
#print(env.observation_space.low)
#> array([-2.4       ,        -inf, -0.20943951,        -inf])

#from gym import spaces
#space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
#x = space.sample()
#assert space.contains(x)
#assert space.n == 8

#from gym import envs
#print(envs.registry.all())

#test for the created Q table

# import gym
# env = gym.make('FrozenLake-v0')
# total_reward = 0
# tt = 0
# ji = 0
# for i_episode in range(10000):
#     observation = env.reset()
#     for t in range(10000):
#         #env.render()
#         #print(observation)
#         action = np.argmax(y[observation][:])
#         observation, reward, done, info = env.step(action)
#         total_reward += reward
#         if t == 99:
#             ji += 1
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             if reward == 1:
#                 print('success!')
#             tt += t+1
#             break
# print(tt/10000)
# print('number of tme step = 100 is ',ji)
# print('percentage of reach limit is',ji/10000)
#print(total_reward/10000)