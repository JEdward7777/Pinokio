import random
import gym, math, json
import numpy as np
import parse_in_dictionary
from gym import spaces

words_file = 'words.json'
sentances_file = "/home/lansford/Sync/projects/tf_over/pinokio/Pinokio/spa-eng/spa.txt"

class SentancePair:
    _input = None
    output = None
    def __init__( self ):
        self._input = []
        self.output = []
    
PUSH_TO = 0
PULL_FROM = 1
OUTPUT = 0
STACK = 1
DIC = 2
INPUT = 3

class Pinokio2(gym.Env):
    nsteps = 0
    
    words = None
    
    #tracking for reward
    starting_sentance_length = 0
    unique_words_pulled_from_stack = None
    unique_words_pulled_from_dict = None
    
    output = None
    stack = None
    _input = None
    accumulator = None
    dictionary = None
    
    sentance_pairs = None
    selected_pair = None
    

    def __init__(self,skip_load=False):
        if skip_load:
            return
        
        self._load_words()
        self._load_sentance_pairs()
        
        #0: which action 0 push 1 pull
        #1: what thing   0 output, 1 stack, 2 dictionary, 3 input
        self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(4))) 
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
            obs.append( self._input[-1] )
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
        done = (not self.output) or (self.output[-1] != self._word_to_index( "<eos>" ))
            
        if not done:
            action_dec = {PUSH_TO:"push to",PULL_FROM:"pull from"}
            what_dec = {OUTPUT:"output",STACK:"stack",DIC:"dic",INPUT:"input"}
            
            print( action_dec[action[0]] + " " + what_dec[action[1]] )
            
            if action[1] == OUTPUT:
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
                    self.dictionary = self.words["index_to_word"][str(self.accumulator)]["dict"]
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
                        self.accumulator = self._input.pop()
                    else:
                        self.accumulator = self._word_to_index( "<eos>" )
                
                
            if done:
                reward += self._grade_sentance()

        
        obs = self._construct_observations()
        
        return obs, reward, done, {} 

    def close(self):
        pass
    
    def _load_words( self ):
        #load the word information
        with open( words_file ) as json_file:
            self.words = json.load(json_file)
            
    def _word_to_index( self, word ):
        return parse_in_dictionary.get_or_make_entry(
            self.words,
            parse_in_dictionary.tame(word)
            )["index"]
        
    
    def _load_sentance_pairs( self ):
        with open( sentances_file, "rt" ) as source_file_object:
            self.sentance_pairs = []
            for line in source_file_object:
                new_pair = SentancePair()
                english, spanish = line.strip().split( "\t" )
                
                for word in spanish.split( " " ):
                    new_pair._input.append( self._word_to_index( word ) )
                    
                for word in english.split(" "):
                    new_pair.output.append( self._word_to_index( word )  )
                    
                self.sentance_pairs.append( new_pair )
                    
        
    
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
        
    def _grade_sentance( self ):
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
        for test_index, test_word in enumerate(self.output):
            this_result = lineCompIndex()
            this_comp_line.append( this_result )
            this_result.errorCount = test_index+1
            this_result.previous = this_comp_line[test_index]
            this_result.state=STATE_PASSING_TEST_OUTPUT
            this_result.content = test_word
            
        for correct_index, correct_word in enumerate(self.selected_pair.output):
            last_comp_line = this_comp_line
            this_comp_line = []
            this_result = lineCompIndex()
            this_comp_line.append( this_result )
            this_result.errorCount = correct_index+1
            this_result.previous = last_comp_line[0]
            this_result.state = STATE_PASSING_CORRECT_OUTPUT
            this_result.content = correct_word
            
            for test_index, test_word in enumerate(self.output):
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
        reward = 1000 * matches
        
        for extra_word in extra_words:
            if extra_word in skipped_outputs:
                reward += 100
                skipped_outputs.remove(extra_word)
                
        return reward
        
        
        
