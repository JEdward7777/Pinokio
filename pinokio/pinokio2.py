#!/usr/bin/python3
import random, os
import gym, math, json
import numpy as np
import parse_in_dictionary
from gym import spaces

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

words_file = 'words_edited2.json'
#sentances_file = "/home/lansford/Sync/projects/tf_over/pinokio/Pinokio/spa-eng/spa.txt"
sentances_file = "../spa-eng/spa_edited.txt"

class SentancePair:
    _input = None
    outputs = None
    def __init__( self ):
        self._input = []
        self.outputs = []
    
NOOP = 0
PUSH_TO = 1
PULL_FROM = 2
OUTPUT = 1
STACK = 2
DIC = 3
INPUT = 4


NUM_PAIRS = 100


action_dec = {NOOP: "nothing", PUSH_TO:"push to",PULL_FROM:"pull from"}
what_dec = {NOOP: "nothing", OUTPUT:"output",STACK:"stack",DIC:"dic",INPUT:"input"}

class Pinokio2(gym.Env):
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
    accumulator = None
    dictionary = None
    
    sentance_pairs = None
    selected_pair = None


    last_actions = None
    
    def clone( self ):
        result = Pinokio2( skip_load=True )
        result.nsteps = self.nsteps
        result.words = self.words
        result.starting_sentance_length = self.starting_sentance_length
        result.unique_words_pulled_from_stack = self.unique_words_pulled_from_stack
        result.unique_words_pulled_from_dict = self.unique_words_pulled_from_dict
        result.unique_words_pushed_to_dict = self.unique_words_pushed_to_dict
        result.output = self.output[:]
        result.stack = self.stack[:]
        result._input = self._input[:]
        result.accumulator = self.accumulator
        result.dictionary = self.dictionary.copy()
        result.sentance_pairs = self.sentance_pairs
        result.selected_pair = self.selected_pair
        return result
    
    def hash_string( self ):
        #return "{}{}{}{}{}{}".format( self.nsteps, self.output, self.stack, self._input, self.accumulator, self.dictionary )
        result = "{}{}{}{}{}".format( self.output, self.stack, self._input, self.accumulator, self.dictionary )
        print( result )
        self.render()
        return result
    
    def translate_word( self, word ):
        return self.words['index_to_word'][str(word)]['word']
    
    def translate_list( self, word_list ):
        return [self.words['index_to_word'][str(word)]['word'] for word in word_list ]  

    def decode_action( self, action=None ):
        if action is None: action = self.last_actions
        if action is None: return "None"
        return action_dec[action[0]] + " " + what_dec[action[1]]
        

    def str_render( self ):
        
        result = ""

        if self.last_actions is not None: result += self.decode_action() + "\n"


        result += str( ("output = [ " + str(self.translate_list( self.output )) + "]").encode('utf8') ) + "\n"
        result += str( ("stack = [ " + str(self.translate_list( self.stack )) + "]").encode('utf8') ) + "\n"
        result += str( ("input = [ " + str(self.translate_list( self._input )) + "]").encode('utf8') ) + "\n"
        result += str( ("accumulator = " + str(self.translate_list( [self.accumulator] ))).encode('utf8')) + "\n"
        result += str( ("dictionary = [ " + str(self.translate_list( self.dictionary )) + "]").encode('utf8') ) + "\n"
        result += str( ("nsteps = [ " + str(self.nsteps) + "]").encode('utf8') ) + "\n"
        
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
        #self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(4))) 
        self.action_space = spaces.MultiDiscrete( [3,5] )
        #0 last output
        #1 top of stack
        #2 current dictionary output
        #3 current input
        #4 accumulator
        #self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.int32)
        #using ten million as max instead of np.inf because the env checker apparently doesn't know how to handle np.inf.
        self.observation_space = spaces.Box(low=0, high=10000000, shape=(5,), dtype=np.int32)
        self.reset()
        
    def _construct_observations( self ):
        obs = []
        #0 last output
        if self.output:
            obs.append( self.output[-1] )
        else:
            obs.append( self._word_to_index( "uh" ) )
            
        #1 top of stack
        if self.stack:
            obs.append( self.stack[-1] )
        else:
            obs.append( self._word_to_index( "uh" ) )
            
        #2 current dictionary output
        if self.dictionary:
            obs.append( self.dictionary[0] )
        else:
            obs.append( self._word_to_index( "uh" ) )
            
        #3 current input
        if self._input:
            obs.append( self._input[0] )
        else:
            obs.append( self._word_to_index( "<eos>" ) )
            
        #4 accumulator
        if self.accumulator:
            obs.append( self.accumulator )
        else:
            obs.append( self._word_to_index( "uh" ) )
        return np.asarray(obs)

    def reset( self ):
        self._pick_next_input()
        self.nsteps = 0
        return self._construct_observations()

    def step(self, action):
        
        self.nsteps += 1

        reward = 0


        done = not not (self.output and self.output[-1] == self._word_to_index( "<eos>" ))

        if self.nsteps > self.starting_sentance_length * 4:
            reward -= (self.nsteps - self.starting_sentance_length) * 10
        
            
        if not done:
            self.last_actions = action
            #action_dec = {NOOP: "nothing", PUSH_TO:"push to",PULL_FROM:"pull from"}
            #what_dec = {NOOP: "nothing", OUTPUT:"output",STACK:"stack",DIC:"dic",INPUT:"input"}
            
            #print( action_dec[action[0]] + " " + what_dec[action[1]] )

            if action[1] == NOOP or action[0] == NOOP:
                reward -= 1e5
            
            elif action[1] == OUTPUT:
                if action[0] == PUSH_TO:
                    self.output.append( self.accumulator )
                    #push to output. One point if consumed words is greater then the output length. -1 point for each push double the length of the input.
                    if self.starting_sentance_length - len(self._input) > len(self.output):
                        reward += 1
                    if len(self.output) > 2*self.starting_sentance_length:
                        reward -= 1
                        
                    #once end of sentance has been pushed we are done.
                    if self.accumulator == self._word_to_index( "<eos>" ):
                        done = True
                elif action[0] == PULL_FROM:
                    if self.output:
                        self.accumulator = self.output[-1]
                    else:
                        self.accumulator = self._word_to_index( "uh" )
                        
            elif action[1] == STACK:
                if action[0] == PUSH_TO:
                    self.stack.append( self.accumulator )
                    #push to stack. No reward. Negative 1 point if stack is longer than input.
                    if len(self.stack) > self.starting_sentance_length:
                        reward -= 1
                elif action[0] == PULL_FROM:
                    if self.stack:
                        self.accumulator = self.stack.pop()
                        #pull from stack. No reward if stack is empty. One point for each unique word per sentance.
                        if self.accumulator not in self.unique_words_pulled_from_stack:
                            self.unique_words_pulled_from_stack.append( self.accumulator )
                            reward += 1
                    else:
                        self.accumulator = self._word_to_index( "uh" )
                        
            elif action[1] == DIC:
                if action[0] == PUSH_TO:
                    self.dictionary = self.words["index_to_word"][str(self.accumulator)]["dict"][:]
                    if not self.accumulator in self.unique_words_pushed_to_dict:
                        self.unique_words_pushed_to_dict.append( self.dictionary )
                    else:
                        #don't want to keep pushing the same word over and over to dictionary.
                        reward -= 1

                elif action[0] == PULL_FROM:
                    if self.dictionary:
                        self.accumulator = self.dictionary.pop(0)
                        #pull from dict. One point for the first pull on each unique word.
                        if self.accumulator not in self.unique_words_pulled_from_dict:
                            self.unique_words_pulled_from_dict.append( self.accumulator )
                            reward += 1
                            
                    else:
                        self.accumulator = self._word_to_index( "uh" )
                        
            
            
            elif action[1] == INPUT:
                if action[0] == PUSH_TO:
                    pass #can't push to input.
                elif action[0] == PULL_FROM:
                    if self._input:
                        #pull from input. No reward if the input is empty. One point if consumed words is less than output length.
                        if self.starting_sentance_length - len(self._input) < len(self.output):
                            reward += 1
                        self.accumulator = self._input.pop(0)
                    else:
                        self.accumulator = self._word_to_index( "<eos>" )
                
                
            #if it has fooled around long enough, just grade the sentance.
            if self.nsteps > self.starting_sentance_length * 20:
                done = True

            if done:
                reward += self._grade_sentance()
        
        obs = self._construct_observations()
        
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
                    
        
    
    def _pick_next_input(self):
        self.selected_pair = random.choice( self.sentance_pairs )
        self._input = self.selected_pair._input[:]
        self.original_input = self.selected_pair._input[:]
        self.starting_sentance_length = len(self._input)
        self.output = []
        
        #probably shouldn't reset the stack, but when the sentaces are random it doesn't make sense to keep it.
        self.stack = []
        self.unique_words_pulled_from_stack = []
        self.accumulator = self._word_to_index( "uh" )
        self.dictionary = []
        self.unique_words_pulled_from_dict = []
        self.unique_words_pushed_to_dict = []
        
        
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
        
        
save_file = "pinokio2.save"
        
def main():

    env = Pinokio2()
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])

    if os.path.exists( save_file ):
        model = PPO2.load( save_file, env=DummyVecEnv([lambda:env]) )
    else:
        model = PPO2(MlpPolicy, env, verbose=1)

    while True:
        #model.learn(total_timesteps=10000)
        model.learn(total_timesteps=100000)

        model.save( save_file )

        obs = env.reset()
        for i in range(10):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                print( "resetting because " + str(done) )
                env.reset()

if __name__ == "__main__":
    main()
