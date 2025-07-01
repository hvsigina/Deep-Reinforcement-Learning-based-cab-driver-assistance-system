# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        
        self.action_space = []
        for i in range(m):
            for j in range(m):
                if i!=j:
                    self.action_space.append((i,j))
        self.action_space.append((0, 0))
        
        self.state_space =  [[x, y, z] for x in range(m) for y in range(t) for z in range(d)] #[city,time,day]
        
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        
        state_encod = [0] * (m + t + d)
        state_encod[int(state[0])] = 1 #encode location
        state_encod[int(m+state[1])] = 1 #encode time
        state_encod[int(m+t+state[2])] = 1 #encode day
        
        return state_encod

    ## Getting number of requests
    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)
       
        if requests >15:
            requests =15
            
        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests)
        actions = [self.action_space[i] for i in possible_actions_index]

        if (0, 0) not in actions:
            actions.append((0,0))
            possible_actions_index.append(20)
            
        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        reward = 0
        if action[0] == 0 and action[1] == 0:
            reward = -C
        else:
            time_1 = Time_matrix[int(action[0]),int(action[1]),int(state[1]),int(state[2])] #time taken from starting location to end location
            time_2 = Time_matrix[int(state[0]),int(action[0]),int(state[1]),int(state[2])] #time taken from current location to start location
            reward = R*time_1 - C*(time_1 + time_2)
        
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        
        loc = state[0]
        time = state[1]
        day = state[2]
        
        if action[0]==0 and action[1]==0: #for (0,0) tuple representing no-ride
            time += 1
            if time == t: #if we enter next day
                time = 0
                day += 1
                if day == d: #if we enter next week
                    day = 0
        
            next_state = (int(loc),int(time),int(day))
        else: #for any other action than (0,0)
            
            #time1+time2
            total_time =  Time_matrix[int(state[0]),int(action[0]),int(state[1]),int(state[2])] + Time_matrix[int(action[0]),int(action[1]),int(state[1]),int(state[2])]
        
            if time+total_time >= t: #we enter next day
                time = t-(time+total_time)
                day+=1
                if day == d:
                    day = 0 #we enter next week
            else: #end of ride we remain on the same day
                time = time+total_time
            
            next_state = (int(action[1]),int(time),int(day))
  
        return next_state

    def test_run(self):
        """
        This fuction can be used to test the environment
        """
        # Loading the time matrix provided
        import operator
        Time_matrix = np.load("TM.npy")
        print("CURRENT STATE: {}".format(self.state_init))

        # Check request at the init state
        requests = self.requests(self.state_init)
        print("REQUESTS: {}".format(requests))

        # compute rewards
        rewards = []
        for req in requests[1]:
            r =  self.reward_func(self.state_init, req, Time_matrix)
            rewards.append(r)
        print("REWARDS: {}".format(rewards))

        new_states = []
        for req in requests[1]:
            s = self.next_state_func(self.state_init, req, Time_matrix)
            new_states.append(s)
        print("NEW POSSIBLE STATES: {}".format(new_states))

        # if we decide the new state based on max reward
        index, max_reward = max(enumerate(rewards), key=operator.itemgetter(1))
        self.state_init = new_states[index]
        print("MAXIMUM REWARD: {}".format(max_reward))
        print ("ACTION : {}".format(requests[1][index]))
        print("NEW STATE: {}".format(self.state_init))
        print("NN INPUT LAYER (ARC-1): {}".format(self.state_encod_arch1(self.state_init)))
        

    def reset(self):
        return self.action_space, self.state_space, self.state_init
