import os, json
import xml.etree.ElementTree as ET

def parse_diciontary( source_xml, target_json ):
    #make the target file if it doesn't already exist
    if not os.path.exists(target_json):
        with open(target_json, 'w'): pass
        data = { "word_to_index": {},
                "index_to_word": {} }
    else:
        #now load the existing file:
        with open( target_json ) as json_file:
            data = json.load(json_file)

    # data = { "word_to_index": {},
    #         "index_to_word": {} }

    #now load the source dictionary
    tree = ET.parse(source_xml)
    root = tree.getroot()

    for word_group in root.findall( './/w' ):
        word = word_group.find( 'c' ).text
        translated_word_node = word_group.find( 'd' ).text
        if translated_word_node != None:
            translated_words = word_group.find( 'd' ).text.split( " " )
        else:
            translated_words = []
        print( "word: " + word + " translated_words = " + str(translated_words) )

        entry = get_or_make_entry( data, word )

        for translated_word in translated_words:
            other_entry = get_or_make_entry( data, translated_word )
            entry['dict'].append( other_entry['index'] )

    #now save it back out.
    with open(target_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

last_index_used = []    
def get_or_make_entry( data, word ):
    #find the last index used if it hasn't been found yet.
    if len( last_index_used ) == 0:
        last_index_used.append(-1)
        for word,index in data['word_to_index'].items():
            if index > last_index_used[0]: last_index_used[0] = index

    #Create the entry if it isn't there already.
    if word not in data['word_to_index']: 
        #need to make it.
        last_index_used[0] += 1
        data['word_to_index'][word] = last_index_used[0]
        data['index_to_word'][last_index_used[0]] = {
            'index': last_index_used[0],
            'word': word,
            'dict': []
        }

    entry = data['index_to_word'][data['word_to_index'][word]]

    return entry



def main():
    source_xml = "/home/lansford/Sync/projects/tf_over/pinokio/es-en.xml"
    target_json = "words.json"
    parse_diciontary( source_xml, target_json )

if __name__ == "__main__":
    main()