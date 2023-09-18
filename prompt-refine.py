# prompt-refine.py 
# prompt-refine customizes and enhances a prompt with document(s) context.

# prompt-refine.py takes as input:
# [1] prompt-path - the full file-path to the <promptname>.txt
#               default is './prompt/prompt.txt'
# [2] corpus-path - the array of paths to the directories containing collections
#               of json files holding metadata about documents possibly
#               relevant to 'refining' the prompt for specific intentions.
#               Currently the format for these documents is based on the 
#               json-files created by the'arxiv-scraper' application:
#               {
#                  'title': title,
#                  'url': result.pdf_url,
#                  'entry_id': result.entry_id, 
#                  'published': result.published.strftime('%m/%d/%y'),
#                  'text': result.summary
#               }
#               default is ['./corpus/'] - all documents in the local corpus 
# [3] context_radius - the integer number of documents/summaries to be selected
#               because their (nlp-filtered text) vector representation is the 
#               closest in semantic metric distance to the vector 
#               representation of the (nlp-filtered) prompt text

# First, prompt-refine creates a simple <promptname>.json file if it does not
# already exist and writes it adjacent to <promtname>.txt.
# The json format is trivial:
# {
#    'text': prompt-text
# }
# Later nlp-filter.py will, if needed, apply several nlp-processing techniques
# such as stemming and stop-word removal, etc. and conversion to remove
# redundant or non-differentiating information from the text. nlp-filter
# then writes the filtered text to a new field '_text'.

# prompt-refine then finds via src/encoder.py (documented elsewhere) the 
# <context_radius> semantically nearest (vector representations of) 
# docs/summaries in the specified corpus, and creates a 'refined' prompt 
# using the <context_radius> document urls as additions to the prompt.

# The refined prompt is written to the 'refinements' field of the prompt
# JSON-file. Refinements is a JSON sub-object with fields consisting
# of corpus_path(s) used for the prompt-refinement. The refinements are
# specified by timesta,mp within the corpus_path field in order to
# accomodate multiple 'variations' of refinements using the same corpus_path.

# Here is an abstract representation of a canonical prompt-file created:
# {
#   text:'...',     # plain-text of prompt
#   _text:'...',    # nlp-filtered text of prompt
#   refinements:{
#     corpus_pathA:{
#       timestamp1:'...',
#       timestamp2:'...'
#     },
#     corpus_pathB:{
#       timestamp1:'...',
#       timestamp2:'...'
#     }
#   }
# }



import sys
import os
import json
import numpy as np
sys.path.insert(0, './src')
import nlp_filter as nlpf
import docs2matrix as d2m
import zero_center_cols
import knn_urls
import prompt_builder as pb


def action(diagnostics:bool=False) -> None:
  
    #defaults
    # tmp !!!!!!!!!!!!!!!   
    #corpus_paths = ['./corpus/physics.hist-ph-test']
    corpus_paths = ['./corpus/physics.hist-ph']

    prompt_fpath = './prompt/prompt.txt'
    context_radius = 3

    # commandline args - overriding defaults
    nargs = len(sys.argv) - 1
    if(nargs == 1):
        corpus_paths = sys.argv[1]
    if(nargs == 2):
        corpus_paths = sys.argv[1]
        prompt_fpath = sys.argv[2]
    if(nargs == 3):
        corpus_paths = sys.argv[1]
        prompt_fpath = sys.argv[2]
        context_radius = sys.srgv[3]

    #defaults and/or possible commandline modifications of defaults
    if(diagnostics):
        print(f'prompt-refine: prompt_fpath = {prompt_fpath}')
        print(f'prompt-refine: corpus_paths = {corpus_paths}')
        print(f'prompt-refine: context_radius = {context_radius}')

    #if <promptname>.json does not exist, create it in the same directory
    #as prompt_fpath, <promptname>.json has the form:
    # {"text": prompt-text, "_text":""}
    if(prompt_fpath.endswith('json')):
        prompt_fpathj = prompt_fpath
        if(diagnostics):
            print(f'\n{prompt_fpath} ends with suffix `json`')
    else:
        prompt_fpathj = os.path.splitext(prompt_fpath)[0] + '.json'
        if(diagnostics):
            print(f'\n{prompt_fpath} does not end with suffix `json`')

        #read prompt text from prompt_fpath
        f = open(prompt_fpath, 'r')
        prompt = f.read()
        if(diagnostics):
            print(f'prompt-refine: prompt = {prompt}')

        #create <promptname>.json file with two fields
        dictionary = {"text": prompt, "_text": "" }
         
        #write dictionary to json-file
        with open(prompt_fpathj, "w") as prompt_json:
            json.dump(dictionary, prompt_json)


    #start processing
    if(diagnostics):
        print(f'prompt-refine: prompt_fpathj is {prompt_fpathj}')
        print(f'prompt-refine: corpus_paths is {corpus_paths}')

    # nlp-filter the prompt and all documents in corpus_paths array of paths
    # nlp_filter returns urls, _texts, a tuple of dictionaries.
    # Each is indexed by filename.
    # urls contains as first element (prompt-file) the path prompt_fpathj
    #      the rest of the elements are the url value read from the
    #      document json-file
    # _texts contains as first element (prompt-file) the nlp_filtered text
    #      value read from the text-field of the prompt json-file
    #      the rest of the elements are the nlp_filtered text-field value 
    #      read from the document json-file
    #      For all json-files read, the nlp_filtered text is written back to
    #      the json-file in a new '_text' field
    #      NOTE: for all json-files, if the _text-field already exists then
    #            the value is read and written to the _texts dictionary, and
    #            no new field '_text' need be created
    urls, _texts = nlpf.action(prompt_fpathj, corpus_paths, diagnostics)
    print('\n\n\n...................back to prompt-refine.py...')
    print('nlp_filter returns urls, _texts\n')


    # obtain document-term matrix A
    A = d2m.action(_texts, diagnostics)
    print('\n\ndocs2matrix returns A')
    # numpy.ndarray
    print(f'type(A) is {type(A)}')


    # For a given numpy matrix, creates and returns a new matrix with each
    # column centered around the column mean, i.e each column element has the
    # mean subtracted.
    # see 'Understanding Complex Datasets' Skillicorn p. 57
    A = zero_center_cols.action(A)
    print(f'\n\nafter mean centering, zero_center_cols returns A')
    # type List[List]
    print(f'type(A) is {type(A)}')


    #Singular Value Decomposition of A as U*S*Vt (V-transpose)
    print('\n\nSVD: A = U*S*Vt where the rows of  U are normalized `semantic` vectors')
    print('np.linalg.svd returns U, S and Vt\n\n')
    U, S, Vt = np.linalg.svd(A, full_matrices=False)


    # svd => if A is NxM doc-term matrix (N doc-rows, M col-terms), then
    # with no dimension reduction, U is NxM, S is MxM and Vt is MxM
    # Corresponding to docK of A (row K of A) is the Kth row of W = U*S
    # Let <w[k,1], w[k,2], ... , w[k,M]> be the elements of the kth row of W.
    # w[k,1] is the coefficient of row-1 (orthonormal) basis element of Vt 
    # w[k,2] is the coefficient of row-2 (orthonormal) basis element of Vt 
    # ...
    # w[k,M] is the coefficient of row-M (orthonormal) basis element of Vt 
    D = np.diag(S)     
    W = np.matmul(U,D)    

    # NOTE:mypy flags error - List[List[float]] does NOT have attr 'shape' 
    # Numpy ndarrays are not correctly type annotated - so ignore error
    if(diagnostics):
        print('matrix2d.shape is given by (rows, columns)')
        print('A.shape is ' + str(A.shape)); 
        print('U.shape is ' + str(U.shape));
        print('D.shape is ' + str(D.shape) + ' - diagonal of square matrix - non-diag els are zeroes');
        print('Vt.shape is ' + str(Vt.shape));

    print('\nsvd details:')
    print('matrix2d.shape is given by (rows, columns)')
    print('A.shape is ' + str(A.shape)); 
    print('U.shape is ' + str(U.shape));
    print('D.shape is ' + str(D.shape) + ' - diagonal of square matrix - non-diag els are zeroes');
    print('Vt.shape is ' + str(Vt.shape));
    print('W.shape is ' + str(W.shape));
 

    # What remains is to select the context_radius closest document vectors
    # (rows of U) to the prompt vector and find the document urls corresponding
    # to these vectors and pre-pending these urls to form a new 'refined' prompt

    # first, find the K (K=context_radius) semantically similar 
    # document-summaries to the prompt
    _urls = knn_urls.action(W, urls, context_radius) 
    print(f'\n\nurls of docs most similar to the initial prompt:\n{_urls}')


    # create a refined_prompt and determine the refined_prompt_fpath
    refined_prompt, refined_prompt_fpath = pb.refine_prompt(prompt_fpath, _urls)
    print(f'\n\ncreated new refined_prompt:\n{refined_prompt}')
    print(f'new refined_prompt_fpath:\n{refined_prompt_fpath}')


    # write refined_prompt to refined_prompt_fpath
    pb.write_prompt(refined_prompt, refined_prompt_fpath)
    print(f'\n\nnew prompt written to = {refined_prompt_fpath}')




if __name__ == "__main__": 
    print('\n\n+++++++++++ prompt-refine +++++++++++++++++++++')
    print("prompt-refine module running as __main__\n")

    #development
    #action(True)
    
    #production
    action(False)
else:
    print("prompt-refine module imported")
