#!/usr/bin/python3
import pinokio5
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
import numpy as np

def main():
    
    env = pinokio5.Pinokio5()
    model = PPO.load( pinokio5.save_file, env=DummyVecEnv([lambda:env]) )
        
    obs = env.reset()
    for i in range(200):
        print( "obs are {}".format( obs ) )
        
        action, _states = model.predict(obs)
        
        print( "action is a {}".format(action) )
        
        obs, reward, done, info = env.step(action)
        env.render()
        print( "obs were {}".format(obs) )
        if done:
            print( "resetting because " + str(done) )
            env.reset()
            
        if input( "press enter" ) == "exit": return


def translate_all_pairs():

    #env = pinokio3.Pinokio3()
    env = pinokio5.Pinokio5()
    model = PPO.load( pinokio5.save_file, env=DummyVecEnv([lambda:env]) )

    with open( "pinokio5_test_output.txt", "wt" ) as fout:
        for pair in env.sentance_pairs:
            print( "next " + str(pair) )
            obs = env.reset(selected_pair=pair)
            done = False
            while not done and env.nsteps < 1000:
                action, _ = model.predict(obs)
                obs, reward, done, _ = env.step(action)
                print( "{} -> {}".format( env.decode_action(), reward ) )

            fout.write( "=====\n")
            fout.write( "input\t" + " ".join(env.translate_list(pair._input)) + "\n" )
            for output in pair.outputs:
                fout.write( "output\t" + " ".join(env.translate_list(output)) + "\n" )
            fout.write( "test out\t" + " ".join(env.translate_list(env.output)) + "\n" )





if __name__ == "__main__":
    #main()
    translate_all_pairs()
