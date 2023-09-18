# prompt_builder.py
# creates a refined prompt using an array of urls of relevant documents,
# and an initial prompt


from typing import List, Tuple
import os



def refine_prompt(prompt_fpath:str, urls:[str]) -> Tuple[str, str]:
    with open(prompt_fpath, 'r') as f:
        prompt_text = f.read()

    urls_text = '' 
    for url in urls:
        urls_text += '\n' + url
    context = f"Using the given URLs as context: {urls_text}, act as an expert in the subject matter referred to by the documents found at the URLs, as well as whatever you can research on your own."
  
    # new prompt
    refined_prompt = f"{context} {prompt_text}"

    # path for new prompt
    refined_prompt_fpath = prompt_fpath.replace('/prompt/', '/refined-prompt/')

    # return created tuple
    return refined_prompt, refined_prompt_fpath



def write_prompt(refined_prompt:str, refined_prompt_fpath:str) -> None:
    with open(refined_prompt_fpath, 'w') as f:
        f.write(refined_prompt)
