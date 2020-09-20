import os,json,parse_in_dictionary

def parse_wordfile( source_file, target_json ):
    #make the target file if it doesn't already exist
    if not os.path.exists(target_json):
        with open(target_json, 'w'): pass
        data = { "word_to_index": {},
                "index_to_word": {} }
    else:
        #now load the existing file:
        with open( target_json ) as json_file:
            data = json.load(json_file)

    #now iterate through the source_file
    with open( source_file, "rt" ) as source_file_object:
        for line in source_file_object:
            english, spanish = line.strip().split( "\t" )
            for word in english.split(" ") + spanish.split( " " ):
                word = parse_in_dictionary.tame(word)
                parse_in_dictionary.get_or_make_entry(data,word)

    #now save it back out.
    with open(target_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



def main():
    source_file = "/home/lansford/Sync/projects/tf_over/pinokio/Pinokio/spa-eng/spa.txt"
    target_json = "words.json"
    parse_wordfile( source_file, target_json )

if __name__ == "__main__":
    main()