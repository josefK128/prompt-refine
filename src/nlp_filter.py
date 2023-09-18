# nlp_filter
# apply nlp techniques to increase uniqueness and significance of text docs


import json
import os
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from typing import Dict, List, Tuple, Pattern


def filter(regex:Pattern, replace:str, s_pf:str, diagnostics:bool=False) -> str:
    #filter each text - eliminate i.e, e.g., A. Taylor
    s:str = re.sub(regex, replace, s_pf)  #filter by regex
    result = re.subn(regex, replace, s_pf)  #filter by regex
    if diagnostics:
        print('filter: detected ' + str(result[1]) + ' anomaly(ies)!!')
        if s != s_pf:
            print(s_pf + '\n\nreplaced by:\n\n' + s)
    return s



def clarify(json_data:Dict, diagnostics) -> str:

    # nlp-filter the contents of the 'text' field
    # create a new field '_texts' in the json_data, and
    # write the expanded json_data back to the json-file
    #_text = 'copied text field: ' + json_data['text']
    s = json_data['text']
       
    #[1] lower case
    s = s.lower()     
    if diagnostics:
        print(f'\nlowercase, s = {s}')


    #[2] eliminate i.e, e.g., A. Taylor
    rs = '\s([a-z,A-Z]\.)+,*'
    regex = re.compile(rs)
    s_pf = s
    s = filter(regex, '', s_pf, diagnostics)
    if diagnostics:
        print(f'\nname initials periods removed. s = {s}')
    

    #[3] eliminate citations of form [34] for example
    rs = '\[\d+\]'
    regex = re.compile(rs)
    s_pf = s
    s = filter(regex, '', s_pf, diagnostics)
    if diagnostics:
        print(f'\nnumbered citations removed. s = {s}')


    #[4] remove instances of string.punctuation 
    punctuation = str.maketrans({key: None for key in string.punctuation})
    s = s.translate(punctuation)
    if diagnostics:
        print(f'\npunctuation removed s = {s}')


    #[5] tokenize each text to a temporary word list for [6],[7]
    #load stopwords
    k = 0
    a = []
    wordlist = word_tokenize(s)
    if diagnostics:
        print(f'\ninitially wordlist = {wordlist}')
     

    #[6] filter out stopwords
    stop_words = stopwords.words('english')
    wordlist = [word for word in wordlist if word not in stop_words]
    if diagnostics:
        #print(f'\nstop_words = {stop_words}')
        print(f'\nafter stop_words removed, wordlist = {wordlist}')

 
    #[7] stem each word in each word list
    porter = PorterStemmer()
    wordlist = [porter.stem(word) for word in wordlist]
    if diagnostics:
        print(f'\nstemming: wordlist = {wordlist}')


    # re-join word lists to dictionary
    _s = ""
    for term in wordlist:
        _s += ' ' + term
    if diagnostics:
        print(f'\nre-joined string _s = {_s}')


    # write the filtered 'text' s to a new '_text' field 
    # of the Dict json_data
    json_data["_text"] = _s


    # return filtered json_data['text']
    return _s




def action(prompt_fpathj:str, corpus_paths:[str], diagnostics:bool=False) -> Tuple[Dict[int,str], Dict[int,str]]:
   
    print('\n\n+++++++++++ nlp_filter +++++++++++++++++++++')
    urls = []
    _texts = []

    # Read the JSON file at prompt_fpathj
    # First extract the contents of the 'url' field and append the url
    # to the List 'urls
    # if a _text field already exists there is noting to be done
    # if a _text field does NOT exist, then nlp filter the contents of the
    # 'text' field and write the filtered result to a new '_text' field,
    # and also append the _text field content to the list '_texts'
    with open(prompt_fpathj) as prompt_file:
        prompt_data = json.load(prompt_file)
        if diagnostics:
            print('\n\n*********************************************')
            print(f'prompt_file = {prompt_file}')
            print(f'prompt_data = {prompt_data}')

        # urls
        urls.append(prompt_fpathj)

        # _texts
        if prompt_data.get('_text') == '':
            _text = clarify(prompt_data, diagnostics)
            _texts.append(_text)
        else:
            _texts.append(prompt_data['_text'])

        if diagnostics:
            print(f"\n\nnlp_filtered prompt filepath = {urls[0]}")
            print(f"nlp_filtered prompt _text = {_text}")



    # Read the JSON files in the corpus_paths directories
    # First extract the contents of the 'url' field and append the url
    # to the List 'urls
    # if a _text field already exists there is noting to be done
    # if a _text field does NOT exist, then nlp filter the contents of the
    # 'text' field and write the filtered result to a new '_text' field,
    # and also append the _text field content to the list '_texts'
    for corpus_path in corpus_paths:
        if diagnostics:
            print('\n\n*********************************************')
            print(f'corpus_path = {corpus_path}')

        for file in os.listdir(corpus_path):
            file_path = os.path.join(corpus_path, file)            
            if diagnostics:
                print(f'\ncorpus file_path = {file_path}')
            with open(file_path, 'r') as json_file:
                json_data = json.load(json_file)
                if diagnostics:
                    print(f'\njson_data = {json_data}')

                # urls
                urls.append(json_data.get('url'))

                # _texts
                if json_data.get('_text') == None:
                    _text = clarify(json_data, diagnostics)
                    _texts.append(_text)
                else:
                    _texts.append(json_data['_text'])

    if diagnostics:
        print('\n\n*********************************************')
        print(f'\nurls = {urls}')
        print(f'\n_texts = {_texts}')

    return urls, _texts



if __name__ == "__main__": 
    print("nlp_filter module running in diagnostics mode as __main__")
 
    # get current directory
    path = os.getcwd()
    #ppath = os.path.abspath(os.path.join(path, os.pardir))
    print("Current Directory", path)

    # development - test
    #action(path + '/prompt/prompt.json', [path + '/corpus/physics.hist-ph-test'], False)
    action(path + '/prompt/prompt.json', [path + '/corpus/physics.hist-ph-test'], True)

else:
    print("nlp_filter module imported")
