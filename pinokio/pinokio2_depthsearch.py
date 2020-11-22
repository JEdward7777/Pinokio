import math
import pinokio2

def breadth_search( root, talk=True ):
    #construct target_sentance with eos on it.
    eos = root._word_to_index( "<eos>" )
    target_sentance = root.selected_pair.output[:]
    if not target_sentance[-1] == eos: target_sentance.append(eos)
    
    
    #hahve a hash of the states from the hash string to the state.
    search_space = {}
    root.prev = None
    #init the bottom from the root state to search from.
    search_space[root.hash_string()] = root
    queue = [root.hash_string()]
    
    #keep track of the best result we have seen from our searching
    best_value = -math.inf
    best_state = None
    
    #keep looping while we have hashes still in the queue.
    found_it = False
    while queue and not found_it:
        pinokio = search_space[queue.pop(0)]
        
        if talk: 
            print( "queue length {}.  Stack length {}".format( len(queue), len(pinokio.stack) ) )
            pinokio.render()
        
        for which_action in [pinokio2.PUSH_TO,pinokio2.PULL_FROM]:
            for what_thing in [pinokio2.OUTPUT,pinokio2.STACK,pinokio2.DIC,pinokio2.INPUT]:
                if which_action == pinokio2.PUSH_TO and what_thing == pinokio2.INPUT:
                    pass
                elif which_action == pinokio2.PUSH_TO and what_thing == pinokio2.STACK and pinokio.stack and pinokio.stack[-1] == pinokio.accumulator:
                    #don't  push to the stack if the value going to be pushed is already there.
                    if talk: print( "boop" )
                    pass
                else:
                    pinokio_copy = pinokio.clone()
                    pinokio_copy.prev = pinokio
                    
                    #obs, value, done, _ = pinokio_copy.step( (which_action,what_thing) )
                    obs, _, done, _ = pinokio_copy.step( (which_action,what_thing) )
                    
                    
                    #compute the value of this state found
                    our_value = pinokio._grade_sentance()
                    #if it isn't done then it isn't as valueable
                    if not done: our_value /= 10
                    #taking more steps to reach the same output isn't as good
                    our_value -= .01 * pinokio.nsteps
                    #if this is the best value found stash it
                    if our_value > best_value:
                        best_value = our_value
                        best_state = pinokio_copy
                        
                        if talk: print( "Found better state with value {} and output {}".format( best_value, pinokio_copy.translate_list(pinokio_copy.output) ) )
                        
                    bad_output = False
                    #we don't need to hunt down trails which have the wrong output
                    for i in range( len( pinokio_copy.output ) ):
                        if i >= len( target_sentance ):
                            bad_output = True
                        elif target_sentance[i] != pinokio_copy.output[i]:
                            bad_output = True
                        
                    
                    #if we didn't finish, then possibly add to the queue
                    if not done and not bad_output:
                        hash_string = pinokio_copy.hash_string()
                        #if we haven't seen it before just add
                        if hash_string not in search_space:
                            queue.append( hash_string )
                            search_space[ hash_string ] =  pinokio_copy
                        else:
                            #otherwise if it is yonger then add it.
                            contender = search_space[ hash_string ]
                            #if we are yonger then we trump
                            if contender.nsteps > pinokio_copy.nsteps:
                                if hash_string not in queue:
                                    queue.append( hash_string )
                                search_space[ hash_string ] = pinokio_copy
                    else:
                        #stop if we found the actuall answer
                        if pinokio_copy.output == target_sentance:
                            #found it!
                            if talk: print( "Found it" )
                            found_it = True
                            break
            if found_it: break
                            
    #now find out what the first best action was.
    route = []
    second_state = best_state
    while second_state.prev and second_state.prev.prev:
        route.append( second_state )
        second_state = second_state.prev
    route.append( second_state )
    route.append( second_state.prev )
    
    #render the route
    if talk: 
        for state in reversed(route): state.render()
    
    
    best_action = second_state.last_actions
    best_output = best_state.output
        
    return (best_action, best_value, best_output)    
        

def depth_search( pinokio, depth, stash={} ):
    
    hash_string = pinokio.hash_string()
    if True:#not hash_string in stash:
        best_value = -math.inf
        best_action = None
        best_output = None
        
        for which_action in [pinokio2.PUSH_TO,pinokio2.PULL_FROM]:
            for what_thing in [pinokio2.OUTPUT,pinokio2.STACK,pinokio2.DIC,pinokio2.INPUT]:
                if which_action == pinokio2.PUSH_TO and what_thing == pinokio2.INPUT:
                    pass
                else:
                    pinokio_copy = pinokio.clone()
                    #obs, value, done, _ = pinokio_copy.step( (which_action,what_thing) )
                    obs, _, done, _ = pinokio_copy.step( (which_action,what_thing) )
                    test_output = pinokio_copy.output
                    
                    
                    if done:
                        value = pinokio._grade_sentance()
                    elif depth <= 0:
                        value = pinokio._grade_sentance()/10
                    else:
                        _, rec_value,test_output = depth_search( pinokio_copy,depth-1, stash )
                        #value += rec_value
                        value = rec_value - .01 #take a bit off so that it doesn't waist moves.
                    
                        
                        
                    if value > best_value:
                        best_value = value
                        best_action = (which_action,what_thing)
                        best_output = test_output
        if not hash_string in stash:
            stash[hash_string] = (best_action, best_value, best_output)
        else:
            if stash[hash_string] != (best_action, best_value, best_output):
                print( "stash problem." )
                stash_best_action, stash_best_value, stash_best_output = stash[hash_string]
                print( "best_action {} {}".format(stash_best_action, best_action ) )
                print( "best_value {} {}".format(stash_best_value, best_value ) )
                print( "best_output {} {}".format( pinokio.translate_list(stash_best_output), pinokio.translate_list(best_output) ) )
            
    
                    
    return stash[hash_string]


MAX_DEPTH = 5
def main():
    env = pinokio2.Pinokio2()
    
    obs = env.reset()
    while True:
        
        action, value, best_output = breadth_search(env,talk=False)
        print( "Targeting output {} with value {}".format( env.translate_list(best_output ), value ) )
        print( "The correct output is {}".format( env.translate_list(env.selected_pair.output) ) )
        
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            print( "resetting because " + str(done) )
            env.reset()
            
        if input( "press enter" ) == "exit": return
    
    
if __name__ == "__main__":
    main()
