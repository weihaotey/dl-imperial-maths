import matplotlib.pyplot as plt
import numpy as np
def run(env, total_episode):
    '''this is a function to update parameters using the policy gradient method (REINFORCE algorithm) and perform
    100 episodes of normal run for the program to see if the average time step is more than 195'''
    total_reward = 0 #to record the total reward after all training
    alpha = 0.01 #learning rate
    max_steps= 0 #the highest average return of the test run after each episode
    parameter = np.random.rand(4)  # random numbers between 0 and 1 (sometimes set to all zeroes)
    totreward = np.arange(0, total_episode) #track the reward return after each episode
    max_time = 250 # maximum allowed time step
    test_run_episode = 100  # total run for the test run after each episode training
    vec = np.arange(0, test_run_episode) #record the trajectory taken for test run that have highest time step returned
    final_p = np.zeros(4) #initialise the final parameters set being output
    for i_episode in range(total_episode):
        observation = env.reset() #reset the environment after each episode
        treward = 0 #initialise total reward
        parameter_delta = np.zeros(4)
        for _ in range(max_time):
            sigmoid = 1/(1+ np.exp(-np.dot(parameter,observation))) #the sigmoid policy implemented in this problem
            action = np.random.choice([1,0],p=[sigmoid,1-sigmoid]) #choose left action with probability sigmoid and right with probability 1 - sigmoid
            if  action == 1:
                parameter_delta += (1 - sigmoid) * observation * treward #update the parameter but not changing the parameter used
            else:
                parameter_delta -= sigmoid * observation *treward
            observation, reward, done, info = env.step(action) #perform the action

            treward += reward #accumulate reward

            if reward == 0 or _ >= max_time-1: #break the loop for null reward or timestep exceed max step
            #if done:
                total_reward += treward #add the reward of each episode
                totreward[i_episode] = treward #record the trajectory
                break

        parameter += alpha * parameter_delta  #update the parameter after each episode
        steps, trajectory = episoderun(env, test_run_episode, parameter) #perform test run on new parameter
        if steps > max_steps:
            maxiter = i_episode #record the episode which max average time step on test run is achieved
            final_p = parameter #record the final parameter
            max_steps = steps #maximum timestep
            vec = trajectory #record the trajctory
            print(steps, maxiter)
        elif steps >= 195: #break the loop if the time step exceed 195
            maxiter = i_episode
            final_p = parameter
            break

    return maxiter , final_p , max_steps, vec, totreward

def episoderun(env, total_episode, parameter):
    '''run the environment using some inputted parameter and according to sigmoid policy to choose the actions'''
    total_reward = 0
    vector = np.arange(0, total_episode)
    for i_episode in range(total_episode):
        observation = env.reset()
        treward = 0
        for _ in range(250):
            sigmoid = 1/(1+ np.exp(-np.dot(parameter,observation)))
            action = np.random.choice([1,0],p=[sigmoid,1-sigmoid])
            observation, reward, done, info = env.step(action)
            #print(reward)
            #env.render() #print the animation if needed

            treward += reward

            if reward == 0 or _ >= 249 :
            #if done:
                vector[i_episode] = treward
                total_reward += treward
                #print(treward)
                break
    average = total_reward/total_episode
    return average, vector

import gym
env = gym.make('CartPole-v0')

max_iteration,final_parameter,max_steps,vec,totreward = run(env,1000)
print(max_iteration,final_parameter,max_steps)
print(totreward)
plt.plot(np.arange(0,100),vec) #plot the trajectory of the best test run with the best parameter
plt.show()

#parameter = [-17.13883256, -47.17388693,  0.18334865,  30.81753493]
#av, vec = episoderun(env,100,parameter)
#print(av)
#print(vec,'\n\n\n')

#total_episode = 1000
# print(parameter, _, maxreward)
# import time
# time.sleep(3)
# ss = np.random.rand(trials,1)
# tt = np.random.rand(trials,1)
# for j in range(trials):
#     s,v = episoderun(env,total_episode,parameter)
#     ss[j] = s
#     tt[j] = max(v)
#     print(s, max(v))
# plt.plot(np.arange(0,total_episode),v)
# plt.show()




