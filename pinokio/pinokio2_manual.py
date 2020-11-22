import pinokio2
import pinokio2_depthsearch

def main():
    env = pinokio2.Pinokio2()
    
    obs = env.reset()
    done = False
    while True:
        print( "The correct output is {}".format( env.translate_list(env.selected_pair.output) ) )
        env.render()
        command = input( "> " )
        print( "------------------" )
        
        which_action = None
        what_thing = None
        
        if "push" in command:
            which_action = pinokio2.PUSH_TO
        elif "pull" in command:
            which_action = pinokio2.PULL_FROM
        else:
            which_action = None
        
        if "output" in command:
            what_thing = pinokio2.OUTPUT
        elif "stack" in command:
            what_thing = pinokio2.STACK
        elif "dictionary" in command:
            what_thing = pinokio2.DIC
        elif "input" in command:
            what_thing = pinokio2.INPUT
        else:
            what_thing = None
            
        if command == "exit": 
            return
        elif command == "grade":
            env._grade_sentance(talk=True)
            
        elif command == "reset":
            env.reset()
            
        elif command == "hash_string":
            print( env.hash_string() )
            
        elif command.startswith("depthsearch"):
            search_depth = pinokio2_depthsearch.MAX_DEPTH
            try:
                arg = command[len("depthsearch"):].strip()
                search_depth = int(arg)
            except ValueError:
                pass
            
            
            action, value, best_output = pinokio2_depthsearch.depth_search(env,search_depth)
            print( "depthsearch best_output of {}".format( env.translate_list(best_output) )  )
            print( "depthsearch best_value of {}".format( value ) )
            env_clone = env.clone()
            env_clone.step(action)
            env_clone.render()
            
        elif command.startswith("breadthsearch"):
            action, value, best_output = pinokio2_depthsearch.breadth_search(env)
            print( "breadth_search best_output of {}".format( env.translate_list(best_output) )  )
            print( "breadth_search best_value of {}".format( value ) )
            env_clone = env.clone()
            env_clone.step(action)
            env_clone.render()
        
        elif which_action and what_thing:
            
            obs, reward, done, info = env.step((which_action,what_thing))
            print( "reward {}".format(reward) )
            
            if done: print( "need to reset" )
            
        else:
            print( "huh?" )
                
    
if __name__ == "__main__":
    main()
