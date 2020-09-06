import random
import gym, math
import numpy as np
from gym import spaces


class Pinokio2(gym.Env):
    nsteps = 0
    first_run = True
    problems = None

    def __init__(self):
        #0: which action 0 push 1 pull
        #1: what thing   0 output, 1 stack, 2 dictionary, 3 input
        self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(4))) 
        #0 last output
        #1 top of stack
        #2 current dictionary output
        #3 current input
        #4 accumulator
        #self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.int32)
        #using ten million as max instead of np.inf because the env checker apparently doesn't know how to handle np.inf.
        self.observation_space = spaces.Box(low=0, high=10000000, shape=(5,), dtype=np.int32)
        self.reset()

    def reset( self ):
        self.nsteps = 0
        return np.asarray([0,0,0,0,0])

    def step(self, action):
        action_dec = {0:"push to",1:"pull from"}
        what_dec = {0:"output",1:"stack",2:"dic",3:"input"}
        
        print( action_dec[action[0]] + " " + what_dec[action[1]] )
        
        #if not np.isnan(action).any():
            #dist = math.sqrt( (action[0]-.2)**2 + (action[1]-.7)**2 )
            #targeting .2, .7
            #reward = 1-dist
            #done = dist == 0 or self.nsteps > 100
            #print( "input is " + str(action) + " dist is " + str( dist ) )
            #print( "step output is is " + str((np.array(0), reward, done, {} )) )
        #else:
            #raise Exception( "nans!")
            #print( "input is " + str(action) )
            #if self.first_run:
                #print( "nan!!" )
            #reward = -1000
            #done = True
        #self.nsteps += 1
        #self.first_run = False
        
        reward = 1
        done = False
        obs = np.asarray([0,0,0,0,0])
        
        return obs, reward, done, {} 

    def close(self):
        pass
