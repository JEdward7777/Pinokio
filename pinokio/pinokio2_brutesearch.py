import math
import pinokio2


def string_steps( state ):
    result = ""
    
    if state.prev is not None: 
        prev_result, step = string_steps( state.prev )
        result += prev_result + "\n"
    else:
        step = 0
    
    step += 1
    result += "{}: {}\n".format( step, state.decode_action() )
    #result += "{}: ".format( step )
    result += state.str_render()
    return result, step

class BreadthSearchResults:
    best_action = None
    best_value = -math.inf
    best_output = None
    num_steps = math.inf
    loop_count = 0
    best_output_state = None
    
    def __str__( self ):
        result, _ = string_steps( self.best_output_state )
        return result
        
    
MAX_LOOPAGE = 10000
    
def breadth_search( root, talk=True ):
    root = root.clone()

    #construct target_sentances with eos on it.
    eos = root._word_to_index( "<eos>" )
    target_sentances = []
    max_sentance_length = 0
    for target_sentance in root.selected_pair.outputs:
        target_sentance = target_sentance[:]
        if target_sentance[-1] != eos: target_sentance.append(eos)
        target_sentances.append( target_sentance )
        if len( target_sentance ) > max_sentance_length: max_sentance_length = len( target_sentance )
    
    
    #have a hash of the states from the hash string to the state.
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
    loop_count = 0
    while queue and not found_it:
        loop_count += 1
        if loop_count > MAX_LOOPAGE: break
    
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
                        
                    
                    #we don't need to hunt down trails which have the wrong output
                    
                    #assume it is bad unless we find one which matches.
                    bad_output = True
                    for target_sentance in target_sentances:
                        #assume the sentance is good
                        matches_this_one = True
                        for i in range( len( pinokio_copy.output ) ):
                            if i >= len(target_sentance):
                                matches_this_one = False
                            elif target_sentance[i] != pinokio_copy.output[i]:
                                matches_this_one = False
                                
                            #don't need to prove it further if we already know it is bad
                            if not matches_this_one: break
                        
                        #if we found a good one stick with it.
                        if matches_this_one:
                            bad_output = False
                            break
                        
                    
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
                        if best_state.output in target_sentances: found_it = True
                        if found_it: break
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
    
    
    best_output = best_state.output
    
    result = BreadthSearchResults()
    result.best_action = second_state.last_actions
    result.best_value = best_value
    result.best_output = best_state.output
    result.num_steps = len( route )
    result.found_it = found_it
    result.loop_count = loop_count
    result.best_output_state = best_state
        
    return result 
        

def main():
    env = pinokio2.Pinokio2()
    
    obs = env.reset()
    while True:
        
        search_result = breadth_search(env,talk=False)
        print( "Targeting output {} with value {} and loop of {}".format( env.translate_list(search_result.best_output ), search_result.best_value, search_result.loop_count ) )
        print( "The correct outputs are {}".format( [env.translate_list(x) for x in env.selected_pair.outputs] ) )
        
        obs, reward, done, info = env.step(search_result.best_action)
        env.render()
        if done:
            print( "resetting because " + str(done) )
            env.reset()
            
        if input( "press enter" ) == "exit": return
    
    
if __name__ == "__main__":
    main()
