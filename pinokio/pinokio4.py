#!/usr/bin/python3
import random, os
import gym, math, json
import numpy as np
import parse_in_dictionary
from gym import spaces

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

words_file = 'words_edited2.json'
#sentances_file = "/home/lansford/Sync/projects/tf_over/pinokio/Pinokio/spa-eng/spa.txt"
sentances_file = "../spa-eng/spa_edited.txt"

class SentancePair:
    _input = None
    outputs = None
    def __init__( self ):
        self._input = []
        self.outputs = []
    def __str__( self ):
        return "{} -> {}".format( self._input, self.outputs )
    
NOOP = 0
PUSH_TO = 1
PULL_FROM = 2
OUTPUT = 1
STACK = 2
DIC = 3
INPUT = 4


NUM_PAIRS = 300


OUTPUT_CONTEXT = 100
INPUT_CONTEXT = 100
STACK_CONTEXT = 100
DICT_CONTEXT = 20

NUM_REGS = 100


action_dec = {NOOP: "nothing", PUSH_TO:"push to",PULL_FROM:"pull from"}
what_dec = {NOOP: "nothing", OUTPUT:"output",STACK:"stack",DIC:"dic",INPUT:"input"}

class Pinokio4(gym.Env):
    nsteps = 0
    
    words = None
    
    #tracking for reward
    starting_sentance_length = 0
    unique_words_pulled_from_stack = None
    unique_words_pulled_from_dict = None
    unique_words_pushed_to_dict = None
    
    output = None
    stack = None
    _input = None
    regs = None
    dictionary = None
    
    sentance_pairs = None
    selected_pair = None


    last_actions = None

    action_history = None

    returned_done = False
    
    def clone( self ):
        result = Pinokio4( skip_load=True )
        result.nsteps = self.nsteps
        result.words = self.words
        result.starting_sentance_length = self.starting_sentance_length
        result.unique_words_pulled_from_stack = self.unique_words_pulled_from_stack
        result.unique_words_pulled_from_dict = self.unique_words_pulled_from_dict
        result.unique_words_pushed_to_dict = self.unique_words_pushed_to_dict
        result.output = self.output[:]
        result.stack = self.stack[:]
        result._input = self._input[:]
        result.regs = self.regs[:]
        result.dictionary = self.dictionary.copy()
        result.sentance_pairs = self.sentance_pairs
        result.selected_pair = self.selected_pair
        result.action_history = self.action_history[:]
        return result
    
    def hash_string( self ):
        result = "{}{}{}{}{}".format( self.output, self.stack, self._input, self.regs, self.dictionary )
        #print( result )
        #self.render()
        return result
    
    def translate_word( self, word ):
        return self.words['index_to_word'][str(word)]['word']
    
    def translate_list( self, word_list ):
        return [self.words['index_to_word'][str(word)]['word'] for word in word_list ]  

    def decode_action( self, action=None ):
        if action is None: action = self.last_actions
        if action is None: return "None"
        return action_dec[action[0]] + " " + what_dec[action[1]] + " with reg " + str( action[2] )
        

    def str_render( self ):
        
        result = ""

        if self.last_actions is not None: result += self.decode_action() + "\n"


        result += str( ("output = [ " + str(self.translate_list( self.output )) + "]").encode('utf8') ) + "\n"
        result += str( ("stack = [ " + str(self.translate_list( self.stack )) + "]").encode('utf8') ) + "\n"
        result += str( ("input = [ " + str(self.translate_list( self._input )) + "]").encode('utf8') ) + "\n"
        result += str( ("regs = [ " + str(self.translate_list( self.regs )) + "]").encode('utf8') ) + "\n"
        result += str( ("dictionary = [ " + str(self.translate_list( self.dictionary )) + "]").encode('utf8') ) + "\n"
        result += str( ("nsteps = [ " + str(self.nsteps) + "]").encode('utf8') ) + "\n"
        if self.returned_done: result += "done\n"
        
        return result
        

    def render( self ):
        print( self.str_render() )

        
    

    def __init__(self,skip_load=False):
        if skip_load:
            return
        
        self._load_words()
        self._load_sentance_pairs(NUM_PAIRS)
        
        #0: which action 0 noop 1 push 2 pull
        #1: what thing   0 noop 1 output, 2 stack, 3 dictionary, 4 input
        #2: to where
        #self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(4))) 
        self.action_space = spaces.MultiDiscrete( [3,5,NUM_REGS] )
        obs = self.reset()
        self.observation_space = spaces.Box(low=0, high=10000000, shape=(len(obs),), dtype=np.int32)
        
        
    def _construct_observations( self ):
        obs = []

            
        #stack
        obs +=  ([self._word_to_index( "uh" )]*STACK_CONTEXT + self.stack)[-STACK_CONTEXT:]
            
        #current dictionary contents
        obs +=  (self.dictionary + [self._word_to_index( "uh" )]*DICT_CONTEXT)[:DICT_CONTEXT]
            
        #regs
        obs += self.regs

        #last action history.  Two each make 20.
        obs += self.action_history

        #outputs obs
        obs +=  ([self._word_to_index( "<eos>" )]*OUTPUT_CONTEXT + self.output)[-OUTPUT_CONTEXT:]

        #input obs
        obs +=  (self._input + [self._word_to_index( "<eos>" )]*INPUT_CONTEXT)[:INPUT_CONTEXT]

        return np.asarray(obs)

    def reset( self, selected_pair=None ):
        self.regs = [ self._word_to_index( "uh" ) ] * NUM_REGS
        self._pick_next_input(selected_pair)
        self.nsteps = 0
        self.action_history = [0]*20
        self.returned_done = False
        self.last_actions = None
        return self._construct_observations()

    def step(self, action):
        
        self.nsteps += 1

        reward = 0

        push_or_pull = action[0]
        target = action[1]
        reg_index = action[2]

        done = not not (self.output and self.output[-1] == self._word_to_index( "<eos>" ))

        #I think this is swamping the other signals.
        #
        # if self.nsteps > self.starting_sentance_length * 4:
        #     reward -= (self.nsteps - self.starting_sentance_length) * 10

        #It might have been needed, but perhaps this is more gental.
        if self.nsteps > self.starting_sentance_length * 5:
            reward -= 5
        
            
        if not done:
            if self.action_history is None: self.action_history = [0]*20
            self.action_history = self.action_history + list(action)
            while len( self.action_history ) > 20:
                self.action_history.pop(0)
            #action_dec = {NOOP: "nothing", PUSH_TO:"push to",PULL_FROM:"pull from"}
            #what_dec = {NOOP: "nothing", OUTPUT:"output",STACK:"stack",DIC:"dic",INPUT:"input"}
            
            #print( action_dec[push_or_pull] + " " + what_dec[target] )

            if target == NOOP or push_or_pull == NOOP:
                reward -= 1e5
            
            elif target == OUTPUT:
                if push_or_pull == PUSH_TO:
                    self.output.append( self.regs[action[2]] )
                    #push to output. One point if consumed words is greater or equal to the output length. -1 point for each push double the length of the input.
                    if self.starting_sentance_length - len(self._input) >= len(self.output):
                        reward += 1
                    if len(self.output) > 2*self.starting_sentance_length:
                        reward -= 1
                        
                    #once end of sentance has been pushed we are done.
                    if self.regs[reg_index] == self._word_to_index( "<eos>" ):
                        done = True
                elif push_or_pull == PULL_FROM:
                    if self.output:
                        self.regs[reg_index] = self.output[-1]
                    else:
                        self.regs[reg_index] = self._word_to_index( "uh" )
                        
            elif target == STACK:
                if push_or_pull == PUSH_TO:
                    self.stack.append( self.regs[reg_index] )
                    #push to stack. No reward. Negative 1 point if stack is longer than input.
                    if len(self.stack) > self.starting_sentance_length:
                        reward -= 1
                elif push_or_pull == PULL_FROM:
                    if self.stack:
                        self.regs[reg_index] = self.stack.pop()
                        #pull from stack. No reward if stack is empty. One point for each unique word per sentance.
                        if self.regs[reg_index] not in self.unique_words_pulled_from_stack:
                            self.unique_words_pulled_from_stack.append( self.regs[reg_index] )
                            reward += 1
                    else:
                        self.regs[reg_index] = self._word_to_index( "uh" )
                        
            elif target == DIC:
                if push_or_pull == PUSH_TO:
                    self.dictionary = self.words["index_to_word"][str(self.regs[reg_index])]["dict"][:]
                    if not self.regs[reg_index] in self.unique_words_pushed_to_dict:
                        self.unique_words_pushed_to_dict.append( self.regs[reg_index] )
                        reward += 1
                    else:
                        #don't want to keep pushing the same word over and over to dictionary.
                        reward -= 1

                elif push_or_pull == PULL_FROM:
                    if self.dictionary:
                        self.regs[reg_index] = self.dictionary.pop(0)
                        #pull from dict. One point for the first pull on each unique word.
                        if self.regs[reg_index] not in self.unique_words_pulled_from_dict:
                            self.unique_words_pulled_from_dict.append( self.regs[reg_index] )
                            reward += 1
                            
                    else:
                        self.regs[reg_index] = self._word_to_index( "uh" )
                        
            
            
            elif target == INPUT:
                if push_or_pull == PUSH_TO:
                    pass #can't push to input.
                elif push_or_pull == PULL_FROM:

                    #if we just pulled from the input then penalize for pulling from the input again without doing something else first.
                    if np.equal( action, self.last_actions ).all():
                        reward -= 5

                    if self._input:
                        #pull from input. No reward if the input is empty. One point if consumed words is less than output length.
                        if self.starting_sentance_length - len(self._input) <= len(self.output):
                            reward += 1
                        self.regs[reg_index] = self._input.pop(0)
                    else:
                        self.regs[reg_index] = self._word_to_index( "<eos>" )
                
                
            #if it has fooled around long enough, just grade the sentance.
            if self.nsteps > self.starting_sentance_length * 20:
                done = True

            if done:
                reward += self._grade_sentance()
        
        obs = self._construct_observations()

        if done: returned_done = True


        self.last_actions = action
        
        return obs, reward, done, {} 

    def close(self):
        pass
    
    def _load_words( self ):
        #load the word information
        with open( words_file, "rt", encoding='utf-8' ) as json_file:
            self.words = json.load(json_file)
            
    def _word_to_index( self, word ):
        return parse_in_dictionary.get_or_make_entry(
            self.words,
            parse_in_dictionary.tame(word)
            )["index"]
        
    
    def _load_sentance_pairs( self, count=-1 ):
        sentance_pairs_dict = {}
        with open( sentances_file, "rt", encoding='utf-8' ) as source_file_object:
            lines_iterated = 1
            for line in source_file_object:
                english, spanish = line.strip().split( "\t" )
                new_input = []
                new_output = []
                
                for word in spanish.split( " " ):
                    new_input.append( self._word_to_index( word ) )
                    
                for word in english.split(" "):
                    new_output.append( self._word_to_index( word )  )
                    
                _hash = ",".join(str(new_input))
                
                if not _hash in sentance_pairs_dict:
                    pair = SentancePair()
                    sentance_pairs_dict[_hash] = pair
                    pair._input = new_input
                else:
                    pair = sentance_pairs_dict[_hash]
                
                pair.outputs.append( new_output )

                if count > 0 and lines_iterated >= count:
                    break
                
                lines_iterated += 1
        self.sentance_pairs = list(sentance_pairs_dict.values())
                    
        
    
    def _pick_next_input(self,selected_pair=None):
        if selected_pair is not None:
            self.selected_pair = selected_pair
        else:
            self.selected_pair = random.choice( self.sentance_pairs )

        self._input = self.selected_pair._input[:]
        self.original_input = self.selected_pair._input[:]
        self.starting_sentance_length = len(self._input)
        self.output = []
        
        #probably shouldn't reset the stack, but when the sentaces are random it doesn't make sense to keep it.
        self.stack = []
        self.unique_words_pulled_from_stack = [self._word_to_index( "uh" )]
        self.regs = [ self._word_to_index( "uh" ) ] * NUM_REGS
        self.dictionary = []
        self.unique_words_pulled_from_dict = []
        self.unique_words_pushed_to_dict = [self._word_to_index( "uh" )]
        
        
    def _grade_sentance( self, talk=False ):
        def grade_possible_output( correct_output ):
            test_output = [x for x in self.output if x != self._word_to_index( "<eos>" ) ]
            
            #K.  I will give a -1 for every word output.  +1000 points for every correct match and then with the list of ommisions and extras +100 points for each extra word which is in the list of ommitions.
            STATE_PASSING_CORRECT_OUTPUT = 0
            STATE_PASSING_TEST_OUTPUT = 1
            STATE_MATCH = 2
            class lineCompIndex(object):
                __slots__ = ['errorCount', 'previous', 'state', 'content' ]
                def __init__( self ):
                    self.errorCount = 0
                    self.previous = None
                    self.state = STATE_MATCH
                    self.content = -1
            
            #This is an init below the first maches of test output words.
            this_comp_line = []
            this_result = lineCompIndex()
            this_comp_line.append( this_result )
            this_result.errorCount = 0
            this_result.previous = None
            this_result.state = STATE_MATCH
            this_result.content = -1
            for test_index, test_word in enumerate(test_output):
                this_result = lineCompIndex()
                this_comp_line.append( this_result )
                this_result.errorCount = test_index+1
                this_result.previous = this_comp_line[test_index]
                this_result.state=STATE_PASSING_TEST_OUTPUT
                this_result.content = test_word
                
            for correct_index, correct_word in enumerate(correct_output):
                last_comp_line = this_comp_line
                this_comp_line = []
                this_result = lineCompIndex()
                this_comp_line.append( this_result )
                this_result.errorCount = correct_index+1
                this_result.previous = last_comp_line[0]
                this_result.state = STATE_PASSING_CORRECT_OUTPUT
                this_result.content = correct_word
                
                for test_index, test_word in enumerate(test_output):
                    this_result = lineCompIndex()
                    this_comp_line.append( this_result )
                    
                    if test_word == correct_word:
                        this_result.previous = last_comp_line[test_index]
                        this_result.errorCount = this_result.previous.errorCount
                        this_result.state = STATE_MATCH
                        this_result.content = test_word
                    else:
                        if last_comp_line[test_index+1].errorCount < this_comp_line[test_index].errorCount:
                            this_result.previous = last_comp_line[test_index+1]
                            this_result.state = STATE_PASSING_CORRECT_OUTPUT
                            this_result.content = correct_word
                        else:
                            this_result.previous = this_comp_line[test_index]
                            this_result.state = STATE_PASSING_TEST_OUTPUT
                            this_result.content = test_word
                        this_result.errorCount = this_result.previous.errorCount + 1
                    
            skipped_outputs = []
            extra_words = []
            matches = 0
            node = this_result
            while node.previous:
                if node.state == STATE_MATCH:
                    matches += 1
                elif node.state == STATE_PASSING_CORRECT_OUTPUT:
                    skipped_outputs.append( node.content )
                else:
                    extra_words.append( node.content )
                node = node.previous
            
            
            reward = 0
            #1000 points for each mach
            reward = 1000 * matches
            if talk: print( "{} matches so {} points".format( matches, reward ) )
            
            #100 points for each extra word which counts.
            for extra_word in extra_words:
                if extra_word in skipped_outputs:
                    reward += 100
                    if talk: print( "\"{}\" correct word in wrong spot 100 more points, total is {}".format( self.translate_word(extra_word), reward ) )
                    skipped_outputs.remove(extra_word)
                else:
                    #-100 points for every extra word which doesn't count
                    reward -= 100
                    if talk: print( "\"{}\" extra uneeded word.  Lost 100 points, total is {}".format( self.translate_word(extra_word), reward ) )
                    
            return reward
        
        best_result = None
        for correct_output in self.selected_pair.outputs:
            correct_output = [x for x in correct_output if x != self._word_to_index( "<eos>" ) ]
            this_result = grade_possible_output( correct_output )
            #print( "this_result {}".format( this_result ) )
            if best_result is None or this_result > best_result:
                best_result = this_result
        #print( "best_result {} self.selected_pair.outputs.length {} self.selected_pair._input {}".format(best_result,len(self.selected_pair.outputs),self.selected_pair._input) )
        return best_result
        
        
save_file = "pinokio4.save"
tb_log_name = "with_registers_1"
        
def main():

    tensorboard_log = "./log"

    env = Pinokio4()
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])

    if os.path.exists( save_file ):
        model = PPO.load( save_file, env=DummyVecEnv([lambda:env]),tensorboard_log=tensorboard_log )
    else:
        model = PPO(MlpPolicy, env, verbose=1,tensorboard_log=tensorboard_log )

    try:
        while True:
            #model.learn(total_timesteps=10000)
            model.learn(total_timesteps=8000000, tb_log_name=tb_log_name)

            model.save( save_file )

            obs = env.reset()
            for i in range(100):
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                env.render()
                if done:
                    print( "resetting because " + str(done) )
                    env.reset()
    except KeyboardInterrupt:
        print( "Saving before exiting..." )
        model.save( save_file )
        print( "k bye" )

if __name__ == "__main__":
    main()
