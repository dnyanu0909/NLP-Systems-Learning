import re

def improved_tokenizer(text: str)->list:
    """
    Improved tokenizer:
    - Lowercases text
    - Removes punctuation
    - Splits by whitespace
    - Handles contractions (e.g., "don't" -> ["do", "not"])
    """

    text = text.lower()

    #handle contractions
    contractions = {
        "don't": "do not",
        "can't": "can not",
        "won't": "will not",
        "i'm": "i am",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "we're": "we are",
        "they're": "they are"
    }

    for key,value in contractions.items():
        text = text.replace(key,value)

    #remove punctuation
    text = re.sub(r"[^\w\s]", '', text)

    tokens = text.split()
    return tokens

if __name__ == "__main__":
    sample_text = "I can't believe it's already 2024! Time flies..."
    
    tokens = improved_tokenizer(sample_text)
    
    print("Input:", sample_text)
    print("Tokens:", tokens)