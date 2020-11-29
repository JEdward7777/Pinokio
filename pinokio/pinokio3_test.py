#!/usr/bin/python3
import pinokio3
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

def main():
    env = pinokio3.Pinokio3()
    model = PPO2.load( pinokio3.save_file, env=DummyVecEnv([lambda:env]) )
    obs = env.reset()
    for i in range(200):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        print( "obs were {}".format(obs) )
        if done:
            print( "resetting because " + str(done) )
            env.reset()
            
        if input( "press enter" ) == "exit": return


if __name__ == "__main__":
    main()
