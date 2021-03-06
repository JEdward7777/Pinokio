import pinokio2_brutesearch
import pinokio2
import os
from stable_baselines3.ppo import MlpPolicy #, MlpLstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th

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

    def reset( self, selected_pair=None ):
        self.last_results = None
        return pinokio2.Pinokio2.reset( self, selected_pair )


    def step(self, action):
        
        before_step_render = self.str_render()
        
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
            #just reward it for the traction it gets so that it doesn't
            #play games of going backwards so that it can be rewarded for
            #going forward again.
            reward =  self.last_results.num_steps - after_results.num_steps

            # if after_results.num_steps < self.last_results.num_steps:
            #     reward = 10#1
            # elif after_results.num_steps == self.last_results.num_steps:
            #     reward = 2
            # else:
            #     reward = 1#-1

        #a little negativity to encorage not waisting time.
        reward -= .01
        
        # if reward > 2:
        #     print( "=====v" )
        #     print( "Before it is" )
        #     print( before_step_render )
        #     print( "Before string of steps:" )
        #     print( str(self.last_results) )
        #     print( "After string of steps:" )
        #     print( str(after_results) )
        #     print( "last_results.num_steps {} after_results.num_steps {} shooting for output {} correct outputs are {}".format( self.last_results.num_steps, after_results.num_steps, str(self.translate_list(after_results.best_output)).encode(), [self.translate_list(x) for x in self.selected_pair.outputs]  ) )
        #     print( "returning reward {} done {}.  Loop count {} action {}".format( reward, done, after_results.loop_count, self.decode_action(action) ) )
        #     self.render()
        #     print( "=====^" )
            
            
        self.last_results = after_results
        return obs, reward, done, info
        



# save_file = "pinokio3_larger.save"
# net_arch = [20, dict(pi=[64, 256, 64], vf=[64, 64])]
# tb_log_name = "larger_1"

# net_arch = [dict(pi=[64, 64], vf=[64, 64])]
# tb_log_name = "normal"
# save_file = "pinokio3.save"

#now changing net_arch as well as providing sentance context to obs.
save_file = "pinokio3_with_context.save"
net_arch = [100, dict(pi=[64, 256, 64], vf=[64, 64])]
tb_log_name = "context_1"
    
def main():

    tensorboard_log = "./log"

    env = Pinokio3()
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])

    if os.path.exists( save_file ):
        model = PPO.load( save_file, env=DummyVecEnv([lambda:env]),tensorboard_log=tensorboard_log )
    else:
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=net_arch)
        model = PPO(MlpPolicy, DummyVecEnv([lambda:env]), verbose=1,tensorboard_log=tensorboard_log)

    #https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                         name_prefix='pinokio3')


    while True:
        model.learn(total_timesteps=15000000, callback=checkpoint_callback, tb_log_name=tb_log_name )

        model.save( save_file )
        print( "saved" )

        obs = env.reset()
        for i in range(20):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            print( "action {} -> reward {}".format( env.decode_action(action), reward ) )
            env.render()
            if done:
                print( "resetting because " + str(done) )
                env.reset()

if __name__ == "__main__":
    main()
