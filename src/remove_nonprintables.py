# remove_nonprintables
# removes non-printable ascii character from input text - returns filtered text 


import string

def action(text: str) -> str:
    """
    Substitute any non-printable ASCII characters (number > 127) with a space character.
    
    Args:
        text (str): The input text string.
    
    Returns:
        str: The modified text string with non-printable ASCII characters substituted.
    """
    printable_chars = string.printable
    modified_text = ""
    
    for char in text:
        if ord(char) > 127 or char not in printable_chars:
            modified_text += ""
        else:
            modified_text += char
    
    return modified_text
