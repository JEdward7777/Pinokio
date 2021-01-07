import pinokio4
#import pinokio2_brutesearch

def main():
    env = pinokio4.Pinokio4()
    
    obs = env.reset()
    done = False
    while True:
        for output in env.selected_pair.outputs:
            print( "The correct output is {}".format( env.translate_list( output ) ) )
        env.render()
        command = input( "> " )
        print( "------------------" )
        
        which_action = None
        what_thing = None
        reg = None
        
        if "push" in command:
            which_action = pinokio4.PUSH_TO
        elif "pull" in command:
            which_action = pinokio4.PULL_FROM
        else:
            which_action = None
        
        if "output" in command:
            what_thing = pinokio4.OUTPUT
        elif "stack" in command:
            what_thing = pinokio4.STACK
        elif "dictionary" in command:
            what_thing = pinokio4.DIC
        elif "input" in command:
            what_thing = pinokio4.INPUT
        else:
            what_thing = None

        for i in range(pinokio4.NUM_REGS):
            if str(i) in command:
                reg = i
            
        if command == "exit": 
            return
        elif command == "grade":
            env._grade_sentance(talk=True)
            
        elif command == "reset":
            env.reset()
            
        elif command == "hash_string":
            print( env.hash_string() )
            
        # elif command.startswith("breadthsearch"):
        #     search_result = pinokio2_brutesearch.breadth_search(env)
        #     print( "breadth_search best_output of {}".format( env.translate_list(search_result.best_output) )  )
        #     print( "breadth_search best_value of {}".format( search_result.best_value ) )
        #     env_clone = env.clone()
        #     env_clone.step(search_result.best_action)
        #     env_clone.render()
        
        elif which_action and what_thing:
            
            obs, reward, done, info = env.step((which_action,what_thing,reg))
            print( "reward {}".format(reward) )
            
            if done: print( "need to reset" )
            
        else:
            print( "huh?" )
                
    
if __name__ == "__main__":
    main()
