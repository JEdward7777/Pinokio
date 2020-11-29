import pinokio2_brutesearch
import pinokio2
import os
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

use_lstm = True

class Pinokio3(pinokio2.Pinokio2):
    last_results = None
    
    simple_cache = None
    
    def __init__( self ):
        pinokio2.Pinokio2.__init__(self)
        self.simple_cache = {}
        
    def brutesearch_cached( self, talk = False ):
        result = None
        hash_str = self.hash_string()
        if not hash_str in self.simple_cache:
            result = pinokio2_brutesearch.breadth_search( self, talk=talk )
            self.simple_cache[hash_str] = result
        else:
            result = self.simple_cache[hash_str]
        return result


    def step(self, action):
        if self.last_results is None:
            #self.last_results = pinokio2_brutesearch.breadth_search( self, talk=False )
            self.last_results = self.brutesearch_cached( talk=False ) 
            
        #now actually take the step
        obs, inner_reward, done, info = pinokio2.Pinokio2.step( self, action )
        
        #now see what it is like after.
        
        #after_results = pinokio2_brutesearch.breadth_search( self, talk=False )
        after_results = self.brutesearch_cached( talk=False ) 
        
        reward = 0
        if done:
            #reward = inner_reward
            reward = self._grade_sentance()
        elif action[0] == 0 or action[1] == 0:
            #don't like doing nothing.
            reward = 0
        elif not after_results.found_it:
            reward = 0#-10
            
            #we must have gotten lost.
            done = True
        else:
            if after_results.num_steps < self.last_results.num_steps:
                reward = 10#1
            elif after_results.num_steps == self.last_results.num_steps:
                reward = 2
            else:
                reward = 1#-1
        self.last_results = after_results
        
        print( "returning reward {} done {}.  Loop count {}".format( reward, done, after_results.loop_count ) )
        return obs, reward, done, info
        
if use_lstm:
    save_file = "pinokio3lstm.save"
else:
    save_file = "pinokio3.save"
    
def main():

    env = Pinokio3()
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])

    if os.path.exists( save_file ):
        if use_lstm:
            model = PPO2.load( save_file, env=SubprocVecEnv([lambda:env,lambda:env.clone(),lambda:env.clone(),lambda:env.clone()]) )
        else:
            model = PPO2.load( save_file, env=DummyVecEnv([lambda:env]) )
    else:
        if use_lstm:
            model = PPO2(MlpLstmPolicy, SubprocVecEnv([lambda:env,lambda:env.clone(),lambda:env.clone(),lambda:env.clone()]), verbose=1)
        else:
            model = PPO2(MlpPolicy, DummyVecEnv([lambda:env]), verbose=1)

    while True:
        model.learn(total_timesteps=1000)

        model.save( save_file )
        print( "potato" )

        #obs = env.reset()
        #for i in range(200):
            #action, _states = model.predict(obs)
            #obs, reward, done, info = env.step(action)
            #env.render()
            #if done:
                #print( "resetting because " + str(done) )
                #env.reset()

if __name__ == "__main__":
    main()
