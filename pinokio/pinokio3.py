import pinokio2_brutesearch
import pinokio2
import os
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

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
        elif not after_results.found_it:
            reward = 0#-10
            
            #we must have gotten lost.
            done = True
        else:
            if after_results.num_steps < self.last_results.num_steps:
                reward = 3#1
            elif after_results.num_steps == self.last_results.num_steps:
                reward = 2
            else:
                reward = 1#-1
        self.last_results = after_results
        
        print( "returning reward {} done {}.  Loop count {}".format( reward, done, after_results.loop_count ) )
        return obs, reward, done, info
        

save_file = "pinokio3.save"
def main():

    env = Pinokio3()
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])

    if os.path.exists( save_file ):
        model = PPO2.load( save_file, env=DummyVecEnv([lambda:env]) )
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
