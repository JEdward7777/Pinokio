import gym, math
import numpy as np
from gym import spaces

class Pinokio(gym.Env):

    nsteps = 0
    first_run = True

    def __init__(self):
        self.action_space = spaces.Box(np.array([-1,-1]), np.array([+1,+1]))
        self.observation_space = spaces.Box(-1, 1, shape=[1])
        self.reset()

    def reset( self ):
        self.nsteps = 0
        return self.step(np.array([0,0]))[0]

    def step(self, action):
        if not np.isnan(action).any():
            dist = math.sqrt( (action[0]-.2)**2 + (action[1]-.7)**2 )
            #targeting .2, .7
            reward = 1-dist
            done = dist == 0 or self.nsteps > 100
            #print( "input is " + str(action) + " dist is " + str( dist ) )
            #print( "step output is is " + str((np.array(0), reward, done, {} )) )
        else:
            raise Exception( "nans!")
            print( "input is " + str(action) )
            if self.first_run:
                print( "nan!!" )
            reward = -1000
            done = True
        self.nsteps += 1
        self.first_run = False
        return np.array(0), reward, done, {} 

    def close(self):
        pass
