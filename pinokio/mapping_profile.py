import parse_in_dictionary, json
import JLDiff
profile_output = "profile.html"
sentances_file = "../spa-eng/spa_edited.txt"
words_file = 'words_edited2.json'
LOAD_LIMIT = 1000

class SentancePair:
    _input = None
    output = None
    def __init__( self ):
        self._input = []
        self.output = []

def load_words():
    #load the word information
    with open( words_file, "rt", encoding='utf-8' ) as json_file:
        return json.load(json_file)

def save_words( words ):
    #now save it back out.
    with open(words_file, 'w', encoding='utf-8') as f:
        json.dump(words, f, ensure_ascii=False, indent=4)

def word_to_index( words, word ):
    word = parse_in_dictionary.tame(word)
    index = parse_in_dictionary.get_or_make_entry(
        words,
        word
        )["index"]
    return index

def load_sentance_pairs( words ):
    sentance_pairs = []
    with open( sentances_file, "rt", encoding='utf-8' ) as source_file_object:
        sentance_pairs = []
        for line in source_file_object:
            new_pair = SentancePair()
            english, spanish = line.strip().split( "\t" )
            
            for word in spanish.split( " " ):
                index = word_to_index( words, word )
                new_pair._input.append( index )
                
            for word in english.split(" "):
                index = word_to_index( words, word )
                new_pair.output.append( index )
                
            sentance_pairs.append( new_pair )

            #for now just some.
            if len(sentance_pairs) >= LOAD_LIMIT: break 
    return sentance_pairs

def get_dict( words, word_num ):
    return words["index_to_word"][str(word_num)]["dict"][:]

def index_to_word( words, word_num ):
    return words['index_to_word'][str(word_num)]['word']

def write_profile():
    words = load_words()
    with open( profile_output, "wt" ) as fout:
        fout.write( "<head>\n")
        fout.write( "<style>\n")
        fout.write( ".pair {\n")
        fout.write( "  background-color: linen;\n")
        fout.write( "  border-style: solid;\n")
        fout.write( "  margin:5px;\n")
        fout.write( "}\n")
        fout.write( ".inputs {\n" )
        fout.write( "  background-color: lightgreen;\n")
        fout.write( "  border-style: solid;\n")
        fout.write( "  margin:5px;\n")
        fout.write( "}\n")
        fout.write( ".outputs {\n" )
        fout.write( "  background-color: lightblue;\n")
        fout.write( "  border-style: solid;\n")
        fout.write( "  margin:5px;\n")
        fout.write( "}\n")
        fout.write( ".dict {\n" )
        fout.write( "  background-color: pink;\n")
        #fout.write( "  border-style: solid;\n")
        fout.write( "  display: inline-block;\n")
        fout.write( "  margin:5px;\n")
        fout.write( "}\n")
        fout.write( ".dict_word {\n" )
        #fout.write( "  background-color: pink;\n")
        #fout.write( "  border-style: solid;\n")
        fout.write( "  display: inline-block;\n")
        fout.write( "  margin:5px;\n")
        fout.write( "}\n")
        fout.write( "</style>\n")
        fout.write( "</head>\n")
        fout.write( "<body>\n")
        for pair in load_sentance_pairs( words ):
            fout.write( "<div class='pair'>\n" )
            fout.write( "  <div class='inputs'>\n")
            fout.write( "    " )
            for word in pair._input:
                fout.write( "<div class='input_word'>{}".format( index_to_word(words, word) ) )

                fout.write( "<div class='dict'>" )
                for dict_word in get_dict( words, word ):
                    fout.write( "<div class='dict_word'>{}</div>".format( index_to_word(words, dict_word)))
                fout.write( "</div>" )

                fout.write( "</div>")
            fout.write( "  </div>\n")

            fout.write( "  <div class='outputs'>\n")
            fout.write( "    " )
            for word in pair.output:
                fout.write( "<div class='output_word'>{}</div>".format( index_to_word(words, word) ) )
            fout.write( "  </div>\n")

            fout.write( "</div>\n" )
        fout.write( "</body>\n")

def find_problems():
    words = load_words()
    for pair_num,pair in enumerate(load_sentance_pairs( words )):
        start_blob = []
        for input_word in pair._input:
            #add the import word just in case it goes right acrost.
            start_blob.append( input_word )
            #now add all the dictionary options
            for dict_word in get_dict( words, input_word ):
                start_blob.append( dict_word )
            
        #diff = JLDiff.compute_diff( start_blob, pair.output )

        if any( o not in start_blob for o in pair.output ):
            print( "Problem in pair_num {}.".format( pair_num ) )
            print( "start_blob: \"" + " ".join( index_to_word(words,w) + ":" + str(w) for w in start_blob) + "\"" )
            print( "output: \"" + " ".join( index_to_word(words,w) + ":" + str(w) for w in pair.output) + "\"" )
            print( "input: \"" + " ".join( index_to_word(words,w) for w in pair._input) + "\"" )
            print( "input: \"" + " ".join( index_to_word(words,w) + ":" + str(w) for w in pair._input) + "\"" )
            
            print( "The following words in the output can't be produced" )
            for o in pair.output:
                #can't skip output words
                if o not in start_blob:
                    print( index_to_word(words, o) + ":" + str(o), end = " " )

            print()

            
            for o in (o for o in pair.output if o not in start_blob):
                if len(pair._input) == 1:
                    code = pair._input[0]
                else:
                    code = -1
                    while code == -1:
                        code = int(input( "\"{}\"?  -1 save.\n> ".format(index_to_word(words, o) + ":" + str(o) )))
                        if code == -1:
                            print( "saving...")
                            save_words( words )
                
                print( "Stuffing {} into {}.".format( index_to_word(words, o) + ":" + str(o), index_to_word( words, code) + ":" + str(code) ))

                words["index_to_word"][str(code)]["dict"].insert(0,o)

            print()
            print()
            
            if pair_num % 10 == 0:
                print( "saving...")
                save_words( words )

        
            #raise Exception( "Not passable")
                

def dump_dictionaries():
    words = load_words()
    for word in words["index_to_word"].values():
        word["dict"].clear()
    save_words(words)

    

def main():
    write_profile()
    find_problems()



if __name__ == "__main__":
    main()
    #dump_dictionaries()