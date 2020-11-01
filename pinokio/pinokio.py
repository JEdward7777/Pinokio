from gensim.models.keyedvectors import KeyedVectors

import random
import gym, math
import numpy as np
from gym import spaces

class Pinokio(gym.Env):

    nsteps = 0
    first_run = True
    problems = None

    def __init__(self):
        self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Box(np.array([-1,-1]), np.array([+1,+1])))) 
        self.observation_space = spaces.Box(-1, 1, shape=[1])
        self.problems = ProblemHandler()
        self.reset()

    def reset( self ):
        self.nsteps = 0
        self.problems.next_problem()
        return self.step(np.array([0,0]))[0]

    def step(self, action):
        if not np.isnan(action).any():
            dist = math.sqrt( (action[0]-.2)**2 + (action[1]-.7)**2 )
            #targeting .2, .7
            reward = 1-dist
            done = dist == 0 or self.nsteps > 100
            #print( "input is " + str(action) + " dist is " + str( dist ) )
            print( "step output is is " + str((np.array(0), reward, done, {} )) )
        else:
            raise Exception( "nans!")
            print( "input is " + str(action) )
            if self.first_run:
                print( "nan!!" )
            reward = -1000
            done = True
        self.nsteps += 1
        self.first_run = False
        return np.array(0), reward, done, {} 

    def close(self):
        pass


class ProblemHandler:
    a_emb = KeyedVectors.load_word2vec_format(
        r"C:\josh\ai\Pinokio\GoogleNews-vectors-negative300.bin", binary=True, limit=100000 )
    b_emb = KeyedVectors.load_word2vec_format(
        r"C:\josh\ai\Pinokio\sbw_vectors.bin", binary=True )

    problems = None

    a_iterator = 0
    b_iterator = 0
    current_problem = 0

    test_answer = None

    def __init__(self):

        def filter( in_words ):
            out_words = []
            for word in in_words:
                word = word.lower()
                if word.endswith( "," ) or word.endswith( "." ) or word.endswith( "!" ) or word.endswith( "?" ):
                    word = word[:-1]
                if word.startswith( "¡" ) or word.startswith( "¿" ):
                    word = word[1:]
                out_words.append( word )
            return out_words

        self.problems = []
        with open( r"C:\josh\ai\Pinokio\spa-eng\spa.txt", "rt", encoding="utf-8", errors='ignore' ) as spa:
            for line in spa:
                a_string, b_string = line.split( "\t" )
                a_words = filter(a_string.split())
                b_words = filter(b_string.split())

                #not sure how to deal with these so just skip it.
                #if "váyase" in b_words: continue

                for word in a_words:
                    if word not in self.a_emb:
                        raise Exception( "a word " + str( word ) + " can't be embedded.  Help." )
                for word in b_words:
                    if word not in self.b_emb:
                        raise Exception( "b word " + str(word) + " can't be embedded.  Help." )

                self.problems.append( (a_words,b_words) )
        self.next_problem()

    def next_problem( self ):
        self.current_problem = random.randint( 0, len( self.problems ) )
        self.a_iterator = 0
        self.b_iterator = 0
        self.test_answer = []

    def read_input_word( self ):
        if self.a_iterator < len( self.problems[self.current_problem][0] ):
            answer = self.a_emb[ self.problems[self.current_problem][0][self.a_iterator] ]
            self.a_iterator += 1
            return answer

    def write_output_word( self, b ):
        self.test_answer.append( b )

    def similarity( self, vec1, vec2 ):
        return np.dot( gensim.matutils.unitvec(vec1), gensim.matutils.unitvec(vec2) )

    def grade( self ):
        closest_matching_index = []
        b_correct = self.problems[self.current_problem][1]

        if len( self.test_answer ) == 0:
            return -100 #at least say something.

        for b_test_index in range( self.test_answer ):
            b_test = self.test_answer[b_test_index]

            total_similarityness = 0
            closest_matching_index = []
        
            closest_value = similarity( b_test, b_correct[0] )
            closest_index = 1

            for b_answer_word_index in range( 1, len( b_correct ) ):
                b_answer_word = b_correct[b_answer_word_index]
                this_similarity = similarity( b_test, b_answer_word )
                if b_test_index == b_answer_word_index:
                    #give a bit of a bump for the correct location so that
                    #if the word is used more then once it can match the right place.
                    this_similarity += .001
                if this_similarity > closest_value:
                    closest_value = this_similarity
                    closest_index = b_answer_word_index

            closest_matching_index.append( closest_index )
            total_similarityness += closest_value

        #now give a bump for each consecutivly correct placed word.
        for index in range(1, len(closest_matching_index) ):
            if closest_matching_index[index-1] + 1 == closest_matching_index[index]:
                total_similarityness += 1

        #now penalize for the wrong number of words.
        total_similarityness -= math.abs( len(self.test_answer) - len( b_correct) )

        #now normalize it by the length of the sentance.
        total_similarityness /= len(closest_matching_index)

        return total_similarityness

            

                