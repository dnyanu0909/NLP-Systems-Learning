import re   # re meaning regular expression for tokenization


def simple_tokenizer(text: str) -> list:
    """
    Basic tokenizer:
    - Lowercases text
    - Removes punctuation
    - Splits by whitespace
    """

    #lowercase the text
    text = text.lower()

    #remove punctutation
    text = re.sub(r'[^\w\s]', '', text)   #\w matches any word character (alphanumeric and underscore), \s matches any whitespace character, and ^ negates the character class, so [^\w\s] matches any character that is not a word character or whitespace.

    #split by whitespaces
    tokens = text.split()

    return tokens

if __name__ == "__main__":   # This block will only execute if this script is run directly, and not when imported as a module in another script.
    sample_text = "I am stressed about money! It's getting worse..."
    
    tokens = simple_tokenizer(sample_text)
    
    print("Input:", sample_text)
    print("Tokens:", tokens)


# Functions (reusable)
# Docstrings (professional code)
# __main__ (modular design)