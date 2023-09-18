# docs2matrix.py
# obtains the nlp_filtered texts List _texts from prompt-refine
# builds a document-term matrix A where documents are the docs in _List
# and terms are the Tf-Idf (term-frequency-inverse document frequency) of
# all stemmed and filtered words in the docs.
# returns A and the original docs dictionary to encoder.py 
#
# the first (0th) row of A is the term expansion of the nlp_filterednprompt
# rows 1-N of A are the term expansions of the nlp_filtered corpus documents 


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List,Dict,Tuple


def action(_texts:[str], diagnostics:bool=False) -> Tuple[List[List[float]]]:

    print('\n\n+++++++++++ docs2matrix +++++++++++++++++++++\n\n')


    #create dictionary of all docs in _texts
    terms:List[str] = []
    for doc in _texts:
        a = doc.split(' ')
        for token in a:
            if not token in terms:
                terms.append(token) 
                if diagnostics == True:
                    print(f'added token {token} to terms List')


    #report number of docs and terms
    print('\n\ndocs.length = ' + str(len(_texts)))
    print('terms.length = ' + str(len(terms)))



     
    # Initialize TfidfVectorizer - create doc-token matrix
    vectorizer = TfidfVectorizer()
    doc_term = vectorizer.fit_transform(_texts)
    print('\n\n doc_term')
    print(doc_term)


    # convert term counts to tfidf frequencies
    tfidf_tokens = vectorizer.get_feature_names_out()
    print('\n\n tfidf_tokens')
    print(tfidf_tokens)
    print(f'number of tfidf_tokens is {str(len(tfidf_tokens))}')


    # pandas DataFrame
    df = pd.DataFrame(data = doc_term.toarray(), columns = tfidf_tokens)
    print('\n\n df:')
    print(df)


    #create matrix At, token-doc matrix, and A, doc-token matrix
    A = df.to_numpy()
    At = A.transpose()
    print('\n\n At:')
    print(At)


    print('\n\nAt is the term-doc matrix derived from _texts')
    print('A is the desired doc-term matrix')
    print('At is better for viewing.')
    print('matrix2d.shape is given by (rows, columns)')
    print('At.shape is ' + str(At.shape));
    print('A.shape is ' + str(A.shape));
    print('Thus - docs2matrix returning A for further svd decomposition')

    return A



if __name__ == "__main__": 
    print("docs2matrix module running in diagnostics mode as __main__")
    action(['this is a simple not nlp-filtered doc for testing'], True)
else:
    print("docs2matrix module imported")
