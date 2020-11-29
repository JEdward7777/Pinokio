#!/usr/bin/python3
import pinokio3
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines import PPO2
import numpy as np

def main():
    
    if pinokio3.use_lstm:
        env = pinokio3.Pinokio3()
        env_vec = DummyVecEnv([lambda:env,lambda:env.clone(),lambda:env.clone(),lambda:env.clone()])
        model = PPO2.load( pinokio3.save_file, env=env_vec )
    else:
        env = pinokio3.Pinokio3()
        model = PPO2.load( pinokio3.save_file, env=DummyVecEnv([lambda:env]) )
        
    obs = env.reset()
    for i in range(200):
        print( "obs are {}".format( obs ) )
        
        if pinokio3.use_lstm:
            action, _states = model.predict(np.stack((obs,obs,obs,obs)))
            action = action[0]
        else:
            action, _states = model.predict(obs)
        
        print( "action is a {}".format(action) )
        
        obs, reward, done, info = env.step(action)
        env.render()
        print( "obs were {}".format(obs) )
        if done:
            print( "resetting because " + str(done) )
            env.reset()
            
        if input( "press enter" ) == "exit": return


if __name__ == "__main__":
    main()
