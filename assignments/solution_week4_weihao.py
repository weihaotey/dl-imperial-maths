import numpy as np
import random
import gym
env = gym.make('FrozenLake-v0')
alpha = 0.1
gamma = 0.99
total_episode = 15000
trials = 10
average = 0

for j in range(trials):
    total_reward = 0 #record the total number of time it each the goal
    action_size = env.action_space.n
    state_size = env.observation_space.n
    Q = np.random.rand(state_size, action_size) #initialize Q table
    min_epsilon = 0.01
    max_epsilon = 1
    epsilon = 1 # initial value of epsilon
    decay_rate = 0.005
    success_rate = np.zeros(150)
    k = 0
    for i_episode in range(total_episode):
        observation = env.reset()

        action = 0
        for t in range(100):
            prvobs = observation  # to record the previous observation (state)
            #prvact = action
            # epsilon-greedy
            y = random.random()  # randomly choose a number between 0 and 1
            if y > epsilon: #with probability of 1- epsilon
                action = np.argmax(Q[observation, :])  # best path according to Q table
            else: #with probability of epsilon
                action = env.action_space.sample()

            #if t != 0:
            #    Q[prvobs, prvact] += alpha * (reward + gamma * (Q[observation,action]) - Q[prvobs, prvact])
            #prvobs = observation  # to record the previous observation (state)
            observation, reward, done, info = env.step(action)
            total_reward += reward

            Q[prvobs, action] += alpha * (reward + gamma * (max(Q[observation, :])) - Q[prvobs, action])

            if done:
                #print("Episode finished after {} timesteps".format(t+1))
                #env.render()
                break
        # update for eponentially decaying epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * i_episode)
        if (i_episode+1)%100 == 0:
            success_rate[k] = total_reward/100
            total_reward = 0
            k += 1
    average += np.mean(success_rate)
    print('success rate: ',np.mean(success_rate)*100,'percent ')
    import matplotlib.pyplot as plt

    plt.plot(np.arange(0, total_episode/100), success_rate)
    plt.show()
    #print(Q)
print('total success rate: ',average/trials,'percent \n')

